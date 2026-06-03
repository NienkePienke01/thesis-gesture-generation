import torch
import torch.nn as nn
import math
from stage_2_models.turn_taking_transformer import TurnTakingTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
from types import SimpleNamespace
from utils import rotation_conversions as rc
from models_dyadit.vqvae_downsampled import TemporalVertexCodecDownsampled
import json

def load_stage_1_checkpoint(model_path, config_path, device):
    with open(config_path, 'r') as f:
        full_config_dict = json.load(f)
    m = full_config_dict["model_params"]
    model = TemporalVertexCodecDownsampled(
        n_vertices=m["nb_joints"],
        latent_dim=m["output_emb_width"],
        categories=m["code_dim"],
        residual_depth=m["depth"],
        downsample_factor=m["downsample_factor"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return m, model


class SeamlessStage2Trainer:
    def __init__(self, args, gpu_id):
        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        m, self.stage1_model = load_stage_1_checkpoint(
            args.stage_1_pretrain.checkpoint_path, 
            args.stage_1_pretrain.config_path, 
            self.device
        )
        self.stage_1_downsample_factor = m["downsample_factor"]
        
        self.model = TurnTakingTransformer(
            vocab_size=m["code_dim"]+1, # + 1 for masking
            d_model=args.model_params.d_model,
            nhead=args.model_params.nhead,
            num_layers=args.model_params.num_layers,
            dim_feedforward=args.model_params.dim_feedforward,
            dropout=args.model_params.dropout,
            max_seq_len=args.model_params.max_seq_len * 4,
            activation=args.model_params.activation
        ).to(self.device)

        self.model = DDP(self.model, device_ids=[gpu_id])
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.training_params.lr_base)
        
        def lr_lambda(epoch):
            if epoch < args.training_params.warmup_epochs:
                return float(epoch + 1) / float(max(1, args.training_params.warmup_epochs))
            elif epoch < args.training_params.decay_epochs:
                progress = float(epoch - args.training_params.warmup_epochs) / float(
                    max(1, args.training_params.decay_epochs - args.training_params.warmup_epochs)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                return 0.01

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        
        self.motion_loss_fn = nn.CrossEntropyLoss()
        self.state_loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _extract_stage1_tokens(self, target_pose_6d):
        return self.stage1_model.encode(target_pose_6d)

    def train_step(self, dict_data, step_percentage):
        self.model.train()
        self.opt.zero_grad()
        
        target_pose = dict_data["pose"].to(self.device, non_blocking=True)
        batch_size, frames = target_pose.shape[0], target_pose.shape[1]
        target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, frames, 43, 3))
        target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, frames, 258)
        
        gt_tokens = self._extract_stage1_tokens(target_pose_6d)
        max_history = self.args.model_params.max_history_frames
        gt_tokens = gt_tokens[:, -max_history:]
        B, T, L = gt_tokens.shape
        gt_tokens_flat = gt_tokens.reshape(B, T * L)
        
        vad_integers = dict_data["conversation_states"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        emotions = dict_data["emotions"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        bert_features = dict_data["bert"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        
        social_relationship = dict_data["social_relationship"].to(self.device, non_blocking=True)
        personality_traits = dict_data["personality_traits"].to(self.device, non_blocking=True)
        
        audio_features = dict_data["audio"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        partner_audio_features = dict_data["partner_audio"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        
        # Interleave frame-level parameters to match token dimension lengths
        vad_flat = vad_integers.repeat_interleave(L, dim=1)
        emotion_flat = emotions.repeat_interleave(L, dim=1)
        bert_flat = bert_features.repeat_interleave(L, dim=1)
        audio_flat = audio_features.repeat_interleave(L, dim=1)
        partner_audio_flat = partner_audio_features.repeat_interleave(L, dim=1)
        
        past_motion_input = gt_tokens_flat[:, :-1].clone()
        vad_input = vad_flat[:, :-1].clone()
        emotion_input = emotion_flat[:, :-1].clone()
        
        audio_input = audio_flat[:, :-1]
        partner_audio_input = partner_audio_flat[:, :-1]
        bert_input = bert_flat[:, :-1]

        # UGM MASKING based on liveges
        max_mask_prob = 0.40
        cosine_mask_prob = max_mask_prob * (0.5 * (1.0 - math.cos(math.pi * step_percentage)))
        
        mask_token_id = self.args.model_params.code_dim # # Index code_dim is the 'MASK'
        token_mask_indices = torch.rand(past_motion_input.shape, device=self.device) < cosine_mask_prob
        past_motion_input[token_mask_indices] = mask_token_id

        # RM MASKING based on liveges
        if torch.rand(1).item() < 0.15:
            rand_val = torch.rand(1).item()
            if rand_val < 0.33:
                vad_input = torch.randint(0, 3, vad_input.shape, device=self.device)
            elif rand_val < 0.66:
                emotion_input = torch.full_like(emotion_input, 5)
            else:
                personality_traits = torch.zeros_like(personality_traits)
        
        target_tokens = gt_tokens_flat[:, 1:]
        target_states = vad_flat[:, 1:]

        pred_logits, pred_state_logits = self.model(
            past_motion_codes=past_motion_input,
            vad_integers=vad_input,
            emotion_categories=emotion_input,
            social_relationship=social_relationship,
            personality_traits=personality_traits,
            audio_features=audio_input,
            partner_audio_features=partner_audio_input,
            bert_features=bert_input
        )
        
        motion_loss = self.motion_loss_fn(
            pred_logits.reshape(-1, pred_logits.size(-1)), 
            target_tokens.reshape(-1)
        )
        state_loss = self.state_loss_fn(pred_state_logits.reshape(-1, pred_state_logits.size(-1)), target_states.reshape(-1))
        
        total_loss = (self.args.loss_weights.cross_entropy * motion_loss) + (self.args.loss_weights.conversation_state_weight * state_loss)
                     
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.training_params.grad_norm)
        self.opt.step()
        
        return {
            "total": total_loss.item(),
            "motion_loss": motion_loss.item(),
            "state_loss": state_loss.item()
        }

    @torch.no_grad()
    def val_step(self, dict_data):
        self.model.eval()
        
        target_pose = dict_data["pose"].to(self.device, non_blocking=True)
        batch_size, frames = target_pose.shape[0], target_pose.shape[1]
        target_pose_matrix = rc.axis_angle_to_matrix(target_pose.reshape(batch_size, frames, 43, 3))
        target_pose_6d = rc.matrix_to_rotation_6d(target_pose_matrix).reshape(batch_size, frames, 258)
        
        gt_tokens = self._extract_stage1_tokens(target_pose_6d)
        max_history = self.args.model_params.max_history_frames
        gt_tokens = gt_tokens[:, -max_history:]
        B, T, L = gt_tokens.shape
        gt_tokens_flat = gt_tokens.reshape(B, T * L)
        
        vad_integers = dict_data["conversation_states"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        emotions = dict_data["emotions"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        bert_features = dict_data["bert"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        
        social_relationship = dict_data["social_relationship"].to(self.device, non_blocking=True)
        personality_traits = dict_data["personality_traits"].to(self.device, non_blocking=True)
        
        audio_features = dict_data["audio"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        partner_audio_features = dict_data["partner_audio"].to(self.device, non_blocking=True)[:, ::self.stage_1_downsample_factor]
        
        vad_flat = vad_integers.repeat_interleave(L, dim=1)
        emotion_flat = emotions.repeat_interleave(L, dim=1)
        bert_flat = bert_features.repeat_interleave(L, dim=1)
        audio_flat = audio_features.repeat_interleave(L, dim=1)
        partner_audio_flat = partner_audio_features.repeat_interleave(L, dim=1)
        
        past_motion_input = gt_tokens_flat[:, :-1]
        vad_input = vad_flat[:, :-1]
        emotion_input = emotion_flat[:, :-1]
        
        audio_input = audio_flat[:, :-1]
        partner_audio_input = partner_audio_flat[:, :-1]
        bert_input = bert_flat[:, :-1]
        
        target_tokens = gt_tokens_flat[:, 1:]
        target_states = vad_flat[:, 1:]
        
        pred_logits, pred_state_logits = self.model(
            past_motion_codes=past_motion_input,
            vad_integers=vad_input,
            emotion_categories=emotion_input,
            social_relationship=social_relationship,
            personality_traits=personality_traits,
            audio_features=audio_input,
            partner_audio_features=partner_audio_input,
            bert_features=bert_input
        )
        
        motion_loss = self.motion_loss_fn(
            pred_logits.reshape(-1, pred_logits.size(-1)), 
            target_tokens.reshape(-1)
        )
        state_loss = self.state_loss_fn(pred_state_logits.reshape(-1, pred_state_logits.size(-1)), target_states.reshape(-1))
        
        total_loss = (self.args.loss_weights.cross_entropy * motion_loss) + (self.args.loss_weights.conversation_state_weight * state_loss)
                     
        return {
            "total": total_loss.item(),
            "motion_loss": motion_loss.item(),
            "state_loss": state_loss.item()
        }