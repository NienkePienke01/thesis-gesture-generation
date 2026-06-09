import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .FiLM_utils import FiLMTransformerDecoderLayer
from .util_classes import PeriodicPositionalEncoding, PredictResidualZQ, MLP, PositionalEncoding
from .audio_utils import setup_audio_model


class AutoRegressiveMotionTransformer(nn.Module):
    def __init__(self, args, vocab_size=512):
        super().__init__()
        d_model = args.model_params.d_model
        # Cfg masking
        self.null_cond_embed = nn.Parameter(torch.randn(1, args.model_params.max_seq_len, d_model)) # For the frame based timeline
        self.null_global_hidden = nn.Parameter(torch.randn(1, d_model))  # For the fused static vector for the film layers
        #
        
        self.motion_embedding = nn.Embedding(vocab_size, d_model)
        self.vad_embedding = nn.Embedding(3, d_model)
        self.emotion_embedding = nn.Embedding(8, d_model)
        self.social_embedding = nn.Embedding(11, d_model)
        self.personality_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=args.model_params.dropout, max_len=args.model_params.max_seq_len)
        # self.pos_encoder = PeriodicPositionalEncoding(d_model, period=args.model_params.stage_1_layers, max_seq_len=args.model_params.max_seq_len)
        
        self.audio_proj = setup_audio_model(args.model_params.audio_size, d_model, args.model_params.audio_layers)
        self.norm_cond = nn.LayerNorm(d_model)
        self.static_audio_cond_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
    
        self.bert_proj = nn.Linear(args.model_params.bert_size, d_model)

        self.memory_fusion = nn.Linear(d_model * 2, d_model)
        self.audio_gate_weights = nn.Embedding(3, 2)

        self.fused_condition_proj = nn.Linear(d_model * 2, d_model)
        
        self.transformer_layers = nn.ModuleList([
            FiLMTransformerDecoderLayer(
                d_model=d_model,
                nhead=args.model_params.nhead,
                dim_feedforward=args.model_params.dim_feedforward,
                dropout=args.model_params.dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(args.model_params.num_layers)
        ])
        
        self.base_classifier = MLP(d_model, d_model, vocab_size)
        self.residual_predictor = PredictResidualZQ(latent_dim=d_model, 
                                                    n_tokens=args.model_params.n_tokens,
                                                    num_head=args.model_params.nhead,
                                                    ffn_dim=args.model_params.dim_feedforward,
                                                    dropout=args.model_params.dropout,
                                                    vocab_size=vocab_size)

    def _pass_audio_layer(self, audio_in):
        audio_in = audio_in.permute(0, 2, 1) # 1D Conv expects: [Batch, Channels, Frames] -> [32, 1024, 106]
        features = self.audio_proj(audio_in)
        features = features.permute(0, 2, 1) # Swap back to [Batch, Frames, d_model]
        return features

    def get_tgt_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)

    def hubert_encoder_block():
        return nn.Sequential(
            nn.Conv1d(1024, 256, 3, 1, 1, bias=False),
            # nn.GroupNorm(32, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, 3, 1, 1, bias=False)
        )

    def apply_cfg_masking(self, global_condition, cond_tokens, cond_drop_prob=0.0):
        if cond_drop_prob == 0.0 or not self.training:
            return global_condition, cond_tokens
            
        batch_size = global_condition.shape[0]
        device = global_condition.device
        
        # Generate one mask per batch item
        keep_mask = torch.rand(batch_size, device=device) >= cond_drop_prob
        
        # Reshape the same mask to match the dimensions of both targets
        keep_mask_film = keep_mask.unsqueeze(-1)                            # Shape: [B, 1]
        keep_mask_cross_attention = keep_mask.unsqueeze(-1).unsqueeze(-1) # Shape: [B, 1, 1]
        
        # Apply to the static global information
        null_hidden = self.null_global_hidden.to(global_condition.dtype)
        masked_global = torch.where(keep_mask_film, global_condition, null_hidden)
        
        # Apply to the timeline sequence
        null_embed = self.null_cond_embed.to(cond_tokens.dtype)
        masked_tokens = torch.where(keep_mask_cross_attention, cond_tokens, null_embed[:, :cond_tokens.shape[1], :])
        
        return masked_global, masked_tokens

    def forward(self, motion_tokens, audio, partner_audio, bert, emotions, social, personality, conversation_state, cond_drop_prob: float = 0.0):

        B, T, D = motion_tokens.shape
        device = motion_tokens.device
        tgt_mask = self.get_tgt_mask(T, device)

        # Motion codes (uses causal mask to look look into future)
        base_motion_tokens = motion_tokens[:, :, 0] # Take only the first code since the main head, will generate it
        motion_embedded = self.motion_embedding(base_motion_tokens)
        motion_embedded = self.pos_encoder(motion_embedded)
 
        # Local timing audio, used to have audio reference for each frame (64 frames) (Cross-Attention)
        conditioned_audio = self._pass_audio_layer(audio)
        conditioned_audio = self.pos_encoder(conditioned_audio) # Local Timing audio

        # Global (static) information
        personality_hidden = self.personality_proj(personality) # global personality (resembles a speakerid)
        # Static audio representation for overall audio intensity
        audio_hidden = conditioned_audio.mean(dim=1)
        audio_hidden = self.static_audio_cond_projection(audio_hidden)

        fused_static_features = torch.cat([personality_hidden, audio_hidden], dim=-1)
    
        global_condition = self.fused_condition_proj(fused_static_features)

        global_condition, cross_attention_condition = self.apply_cfg_masking(global_condition, conditioned_audio)

        # Layer normalisation (based on photo2real)
        cross_attention_condition = self.norm_cond(cross_attention_condition)

        
        # gating_weights = torch.softmax(self.audio_gate_weights(conversation_state), dim=-1)
        # memory = self.memory_fusion(torch.cat([(self.audio_proj(audio) * gating_weights[:, :, 0:1]) + 
        #                                        (self.audio_proj(partner_audio) * gating_weights[:, :, 1:2]),
        #                                          self.bert_proj(bert)
        #                                          ], dim=-1))

        # vad_16 = conversation_state.view(B, T+1, D).mode(dim=2).values # grabs the most frequent value of each group (splits the 64 frames into groups of 4)
        # vad_input = vad_16[:, :-1].long()

        # emotion_16 = emotions.view(B, T+1, D).mode(dim=2).values # Grabs the mean of each group (64 frames into groups of 4)
        # emotion_input = emotion_16[:, :-1].long()

        # base_motion_tokens = motion_tokens[:, :, 0] # Take only the first code since the main head, will generate it
        # motion_embedded = self.motion_embedding(base_motion_tokens)
        
        # x = self.pos_encoder(motion_embedded + self.vad_embedding(vad_input) + 
        #                      self.emotion_embedding(emotion_input) + 
        #                      self.personality_proj(personality).unsqueeze(1).repeat(1, T, 1) + 
        #                      self.social_embedding(social).unsqueeze(1).repeat(1, T, 1))
        
        features = motion_embedded
        for layer in self.transformer_layers:
            features = layer(
                tgt=features,
                memory=cross_attention_condition,
                t=global_condition,
                tgt_mask=tgt_mask
            )
        
        base_logits = self.base_classifier(features)
        residual_outputs = list(self.residual_predictor(features, cross_attention_condition))
        zqs = residual_outputs[:self.residual_predictor.n_tokens]
        residual_logits = residual_outputs[self.residual_predictor.n_tokens:]
        
        logits = torch.stack([base_logits] + residual_logits, dim=-1)
        return logits, zqs