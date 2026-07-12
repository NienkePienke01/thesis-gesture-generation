import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import rotation_conversions as rc
from models_dyadit.vqvae_downsampled import TemporalVertexCodecDownsampled
from stage_2_models.autoregressive_transformer import AutoRegressiveMotionTransformer
import json
import math
import random
import torch.nn.functional as F

def load_stage_1_checkpoint(model_path, config_path, device):
    with open(config_path, 'r') as file_reader:
        full_config_dict = json.load(file_reader)
    model_parameters = full_config_dict["model_params"]
    model = TemporalVertexCodecDownsampled(
        n_vertices=model_parameters["nb_joints"],
        latent_dim=model_parameters["output_emb_width"],
        categories=model_parameters["code_dim"],
        residual_depth=model_parameters["depth"],
        downsample_factor=model_parameters["downsample_factor"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model_parameters, model

class SeamlessStage2Trainer:
    def __init__(self, args, gpu_id):
        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        model_parameters, self.stage1_model = load_stage_1_checkpoint(
            args.stage_1_pretrain.checkpoint_path, 
            args.stage_1_pretrain.config_path, 
            self.device
        )
        self.code_dim = model_parameters["code_dim"]
        self.cfg_drop_prob = args.training_params.cfg_drop_prob
        self.model = AutoRegressiveMotionTransformer(args, vocab_size=model_parameters["code_dim"] + 1).to(self.device)
        self.model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.training_params.lr_base, weight_decay=1e-2)
        
        def learning_rate_lambda(epoch):
            minimum_learning_rate_ratio = getattr(args.training_params, "min_lr_ratio", 0.05)
            if epoch < args.training_params.warmup_epochs:
                return float(epoch + 1) / float(max(1, args.training_params.warmup_epochs))
            elif epoch < args.training_params.decay_epochs:
                progress = float(epoch - args.training_params.warmup_epochs) / float(
                    max(1, args.training_params.decay_epochs - args.training_params.warmup_epochs)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return minimum_learning_rate_ratio + (1.0 - minimum_learning_rate_ratio) * cosine_decay
            else:
                return minimum_learning_rate_ratio

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, learning_rate_lambda)
        self.ce_loss_function = nn.CrossEntropyLoss()
        self.mse_loss_function = nn.MSELoss()
        self.soft_labels = self._precompute_soft_labels(temperature=args.training_params.soft_label_temperature)

    @torch.no_grad()
    def _extract_stage1_tokens(self, target_pose_6d):
        return self.stage1_model.encode(target_pose_6d)

    @torch.no_grad()
    def _extract_codebook_latent_vector(self, encoded: torch.Tensor) -> torch.Tensor:
        """tokens -> continuous codebook embedding [B, T // downsample_factor, latent_dim]."""
        reformat = encoded.dim() > 2
        if reformat:
            B, T_down, _ = encoded.shape
            encoded = encoded.reshape((-1, self.stage1_model.residual_depth))
        rep = self.stage1_model.quantizer.decode(encoded)
        if reformat:
            rep = rep.reshape((B, T_down, -1))
        return rep

    @torch.no_grad()
    def _compute_predicted_codebook_latent_vector(self, logits: torch.Tensor) -> torch.Tensor:
        predicted_latent_out = 0
        for i, layer in enumerate(self.stage1_model.quantizer.layers):
            probs = nn.functional.softmax(logits[:, :, :, i], dim=-1) # Probability for each code in this layer [B, T, 512]

            mask_embedding = torch.zeros(1, layer.codebook.shape[1], device=layer.codebook.device) # Additional vector of for example size 64 to add (0s) since its additive
            codebook_layer = torch.cat([layer.codebook, mask_embedding], dim=0) # layer.codebook is [513, 64] where 512(+1) = codedim with mask and 64 the latent vector size
            
            soft_quantized = torch.matmul(probs, codebook_layer)
            predicted_latent_out += soft_quantized

        return predicted_latent_out

    @torch.no_grad()
    def _precompute_soft_labels(self, temperature=0.05):
        soft_label_matrices = []
        
        for layer in self.stage1_model.quantizer.layers:
            dist_matrix = torch.cdist(layer.codebook, layer.codebook, p=2) # pairwise Euclidean distances [Code_dim, Code_dim]
            # -dist so closer items get higher probability
            soft_labels = F.softmax(-dist_matrix / temperature, dim=-1) # Convert to probabilities using Temperature Softmax

            zero_column = torch.zeros(soft_labels.shape[0], 1, device=soft_labels.device) # extra, for the mask token
            soft_labels = torch.cat([soft_labels, zero_column], dim=-1)

            dummy_row = torch.zeros(1, soft_labels.shape[1], device=soft_labels.device) # dummy row of zeros at the bottom to prevent out-of-bounds
            soft_labels = torch.cat([soft_labels, dummy_row], dim=0)
            
            soft_label_matrices.append(soft_labels)
            
        return soft_label_matrices

    def _mask_ground_truth_tokens(self, max_mask_prob, step_percentage, input_tokens, mask_token):
        """
        Masks the input tokens with the mask_token, will mask in blocks so not uniform random
        Will never mask the first motion code out of all layers, therefore the first head will always be trained with trainer enforcing
        """
        if max_mask_prob == 0:
            return input_tokens.clone()
        B, T, L = input_tokens.shape
        device = input_tokens.device
        cosine_mask_prob = max_mask_prob * (0.5 * (1.0 - math.cos(math.pi * step_percentage)))
        mask = torch.zeros((B, T, L), dtype=torch.bool, device=device)

        max_block_size = L-1 # Block masking, define the max block size
        block_size = max(1, math.ceil(max_block_size * step_percentage))
        num_blocks = max(1, int((T * max_block_size * cosine_mask_prob) / block_size)) # Calculate how many blocks to mask based on probability
        
        for b in range(B):
            for _ in range(num_blocks):
                start = random.randint(0, T - block_size)
                # Mask the chunk
                num_layers_to_mask = random.randint(1, L-1)
                random_layers = random.sample(range(1, L), num_layers_to_mask)
                mask[b, start : start + block_size, random_layers] = True
                
        # Force Layer 0 (Base) to always be visible to provide semantic stability
        mask[:, :, 0] = False 
        
        # Apply Mask
        masked_tokens = input_tokens.clone()
        masked_tokens[mask] = mask_token
        return masked_tokens, mask

    def train_step(self, dict_data, step_percentage):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Data setup for forward pass
        target_pose = dict_data["pose"].to(self.device, non_blocking=True)
        batch_size, total_frames = target_pose.shape[0], target_pose.shape[1]
        target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, total_frames, 43, 3))
        target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, total_frames, 258)
        
        ground_truth_tokens = self._extract_stage1_tokens(target_pose_6d) # torch.Size([32, 16, 4])
        
        input_tokens = ground_truth_tokens[:, :-1] # torch.Size([32, 15, 4]) tokens 0-14
        #
        target_tokens = ground_truth_tokens[:, 1:] # torch.Size([32, 15, 4]) tokens 1-15 (the ones we want to learn)
        target_latent_vectors = self._extract_codebook_latent_vector(target_tokens)

        #
        max_scheduled_sampling_prob = getattr(self.args.training_params, "scheduled_sampling_prob", 0.5)
        current_scheduled_sampling_prob = max_scheduled_sampling_prob * step_percentage

        # Schedules sampling forward pass, the idea is we do a forward pass to know our predictions, to then do it again, with some of our predicted tokens thrown in
        mixed_active_layer0 = target_tokens[:, :, 0]
        if current_scheduled_sampling_prob > 0:
            with torch.no_grad():
                self.model.eval()
                first_pass_logits, _, _, _ = self.model(
                    motion_tokens=input_tokens,
                    audio=dict_data["wav2vec_audio"].to(self.device),
                    partner_audio = dict_data["partner_wav2vec_audio"].to(self.device) if dict_data.get("partner_wav2vec_audio") is not None else None,
                    bert=dict_data["bert"].to(self.device),
                    emotions=dict_data["emotions"].to(self.device),
                    social = dict_data["social_relationship"].to(self.device) if dict_data.get("social_relationship") is not None else None,
                    personality=dict_data["personality_traits"].to(self.device),
                    conversation_state=dict_data["conversation_states"].to(self.device),
                    cond_drop_prob=0.0,
                    skip_residuals=True # only predict layer 0
                )
                predicted_layer0 = torch.argmax(first_pass_logits, dim=-1)
                self.model.train()

            B, T, L = input_tokens.shape

            # we mix the Input History (Frames 1-14), so some of the tokens are predicted instead of GT
            # also shift predicted_targets back by 1 ([:-1]) to align with input_tokens history ([1:])
            replace_mask_inputs = torch.rand((B, T - 1), device=self.device) < current_scheduled_sampling_prob
            mixed_input_tokens = input_tokens.clone()
            mixed_input_tokens[:, 1:, 0] = torch.where(
                replace_mask_inputs, 
                predicted_layer0[:, :-1], 
                input_tokens[:, 1:, 0]
            )
            input_tokens = mixed_input_tokens
            
            # Mix the Residual Target Layer 0
            # Force the residual heads to occasionally learn from predicted tokens instead of ground truth
            replace_mask_residual = torch.rand((B, T), device=self.device) < current_scheduled_sampling_prob
            mixed_active_layer0 = torch.where(
                replace_mask_residual, 
                predicted_layer0, 
                target_tokens[:, :, 0]
            )

        # Apply the token mask to sometimes completely hide tokens and make the model predict without token features: set max_mask_prob to 0 to skip this
        # input_tokens, used_mask = self._mask_ground_truth_tokens(self.args.training_params.max_mask_prob, step_percentage, input_tokens, self.code_dim)

        # Forward pass
        logits, _, _, _ = self.model(
            motion_tokens=input_tokens,
            audio=dict_data["wav2vec_audio"].to(self.device), # torch.Size([32, 64, 768])
            partner_audio = dict_data["partner_wav2vec_audio"].to(self.device) if dict_data.get("partner_wav2vec_audio") is not None else None,
            bert=dict_data["bert"].to(self.device),
            emotions=dict_data["emotions"].to(self.device),
            social = dict_data["social_relationship"].to(self.device) if dict_data.get("social_relationship") is not None else None,
            personality=dict_data["personality_traits"].to(self.device),
            conversation_state=dict_data["conversation_states"].to(self.device),
            cond_drop_prob=self.cfg_drop_prob,
            skip_residuals=False,
            active_layer0_tokens=mixed_active_layer0 # We use predicted tokens and ground truth tokens (a mix)
        )
        predicted_latent_vectors = self._compute_predicted_codebook_latent_vector(logits)

        # Loss calculations
        ce_loss = 0
        masked_loss = 0
        ce_losses_layered = {}

        for i in range(logits.shape[-1]):
            layer_num = i
            layer_display_num = i + 1
            
            # Reshape for soft label calculation
            logits_flat = logits[:, :, :, i].reshape(-1, logits.size(2)) 
            targets_flat = target_tokens[:, :, i].reshape(-1) 
            # mask_flat = used_mask[:, :, i].reshape(-1) 
            
            # get soft label distribution [B*T, vocab_size]
            target_dist = self.soft_labels[layer_num][targets_flat]
            
            # compute log probabilities & soft cross entropy loss
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss_all = -(target_dist * log_probs).sum(dim=-1).mean()
            
            # update tracking
            ce_loss += loss_all
            ce_losses_layered[layer_display_num] = loss_all.item()

            # masked loss (only for masked tokens)
            # if mask_flat.any():
            #     masked_target_dist = target_dist[mask_flat]
            #     masked_log_probs = log_probs[mask_flat]
            #     masked_loss += -(masked_target_dist * masked_log_probs).sum(dim=-1).mean()
        
        ce_loss = ce_loss / logits.shape[-1]
        # masked_loss = masked_loss / logits.shape[-1]

        mse_loss = self.mse_loss_function(target_latent_vectors, predicted_latent_vectors)
            
        # total_loss = ((ce_loss + masked_loss) * self.args.loss_weights.ce_weight) + (self.args.loss_weights.mse_weight * mse_loss)
        total_loss = ((ce_loss) * self.args.loss_weights.ce_weight) + (self.args.loss_weights.mse_weight * mse_loss)
                     
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.training_params.grad_norm)
        self.optimizer.step()
        
        return {"total": total_loss.item(), "ce": ce_loss, "mse": mse_loss, "ce_loss_layered": ce_losses_layered, "masked_loss": masked_loss}

    @torch.no_grad()
    def val_step(self, dict_data):
        self.model.eval()
        
        target_pose = dict_data["pose"].to(self.device, non_blocking=True)
        batch_size, total_frames = target_pose.shape[0], target_pose.shape[1]
        target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, total_frames, 43, 3))
        target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, total_frames, 258)
        
        ground_truth_tokens = self._extract_stage1_tokens(target_pose_6d)
        
        input_tokens = ground_truth_tokens[:, :-1] # torch.Size([32, 15, 4])
        target_tokens = ground_truth_tokens[:, 1:] # torch.Size([32, 15, 4])
        target_latent_vectors = self._extract_codebook_latent_vector(target_tokens)
        
        # pass to get the layer 0 predictions
        first_pass_logits, _, _, _ = self.model(
            motion_tokens=input_tokens,
            audio=dict_data["wav2vec_audio"].to(self.device),
            partner_audio = dict_data["partner_wav2vec_audio"].to(self.device) if dict_data.get("partner_wav2vec_audio") is not None else None,
            bert=dict_data["bert"].to(self.device),
            emotions=dict_data["emotions"].to(self.device),
            social = dict_data["social_relationship"].to(self.device) if dict_data.get("social_relationship") is not None else None,
            personality=dict_data["personality_traits"].to(self.device),
            conversation_state=dict_data["conversation_states"].to(self.device),
            cond_drop_prob=0.0,
            skip_residuals=True
        )
        first_pass_logits[..., self.code_dim] = float('-inf') # block mask token
        predicted_layer0 = torch.argmax(first_pass_logits, dim=-1)

        simulated_input_tokens = input_tokens.clone()
        simulated_input_tokens[:, 1:, 0] = predicted_layer0[:, :-1]
        simulated_active_layer0 = predicted_layer0

        # Pass to get the residual layers based on our predicted layer 0
        logits, _, _, _ = self.model(
            motion_tokens=simulated_input_tokens,
            audio=dict_data["wav2vec_audio"].to(self.device),
            partner_audio = dict_data["partner_wav2vec_audio"].to(self.device) if dict_data.get("partner_wav2vec_audio") is not None else None,
            bert=dict_data["bert"].to(self.device),
            emotions=dict_data["emotions"].to(self.device),
            social = dict_data["social_relationship"].to(self.device) if dict_data.get("social_relationship") is not None else None,
            personality=dict_data["personality_traits"].to(self.device),
            conversation_state=dict_data["conversation_states"].to(self.device),
            cond_drop_prob=0.0,
            skip_residuals=False,
            active_layer0_tokens=simulated_active_layer0
        )
        predicted_latent_vectors = self._compute_predicted_codebook_latent_vector(logits)
        
        ce_loss = 0
        ce_losses_layered = {}

        for i in range(logits.shape[-1]):
            layer_num = i
            layer_display_num = i + 1
            
            logits_flat = logits[:, :, :, i].reshape(-1, logits.size(2)) 
            targets_flat = target_tokens[:, :, i].reshape(-1) 
            
            target_dist = self.soft_labels[layer_num][targets_flat]
            
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss_all = -(target_dist * log_probs).sum(dim=-1).mean()
            
            ce_loss += loss_all
            ce_losses_layered[layer_display_num] = loss_all.item()

        ce_loss = ce_loss / logits.shape[-1]
        mse_loss = self.mse_loss_function(target_latent_vectors, predicted_latent_vectors)
        
        total_loss = (ce_loss * self.args.loss_weights.ce_weight) + (self.args.loss_weights.mse_weight * mse_loss)
                     
        return {"total": total_loss.item(), "ce": ce_loss, "mse": mse_loss, "ce_loss_layered": ce_losses_layered}