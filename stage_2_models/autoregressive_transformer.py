import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .FiLM_utils import FiLMTransformerDecoderLayer
from .util_classes import PeriodicPositionalEncoding, PredictResidualZQ, MLP
from .audio_utils import setup_audio_model


class AutoRegressiveMotionTransformer(nn.Module):
    def __init__(self, args, vocab_size=512):
        super().__init__()
        d_model = args.model_params.d_model
        
        self.motion_embedding = nn.Embedding(vocab_size, d_model)
        self.vad_embedding = nn.Embedding(3, d_model)
        self.emotion_embedding = nn.Embedding(8, d_model)
        self.social_embedding = nn.Embedding(11, d_model)
        self.personality_proj = nn.Linear(5, d_model)
        self.pos_encoder = PeriodicPositionalEncoding(d_model)
        
        self.audio_proj = setup_audio_model()
        self.bert_proj = nn.Linear(768, d_model)
        self.memory_fusion = nn.Linear(d_model * 2, d_model)
        self.audio_gate_weights = nn.Embedding(3, 2)
        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=args.model_params.nhead, 
                                       dim_feedforward=args.model_params.dim_feedforward, batch_first=True), 
            num_layers=args.model_params.num_layers
        )
        
        self.base_classifier = MLP(d_model, d_model, vocab_size)
        self.residual_predictor = PredictResidualZQ(latent_dim=d_model, 
                                                    n_tokens=args.model_params.n_tokens,
                                                    num_head=args.model_params.nhead,
                                                    ffn_dim=args.model_params.dim_feedforward,
                                                    dropout=args.model_params.dropout,
                                                    vocab_size=vocab_size)

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

    def forward(self, motion_tokens, audio, partner_audio, bert, emotions, social, personality, conversation_state):

        B, T, D = motion_tokens.shape
        device = motion_tokens.device

        
        # gating_weights = torch.softmax(self.audio_gate_weights(conversation_state), dim=-1)
        # memory = self.memory_fusion(torch.cat([(self.audio_proj(audio) * gating_weights[:, :, 0:1]) + 
        #                                        (self.audio_proj(partner_audio) * gating_weights[:, :, 1:2]),
        #                                          self.bert_proj(bert)
        #                                          ], dim=-1))

        vad_16 = conversation_state.view(B, T+1, D).mode(dim=2).values # grabs the most frequent value of each group (splits the 64 frames into groups of 4)
        vad_input = vad_16[:, :-1].long()

        emotion_16 = emotions.view(B, T+1, D).mode(dim=2).values # Grabs the mean of each group (64 frames into groups of 4)
        emotion_input = emotion_16[:, :-1].long()

        base_motion_tokens = motion_tokens[:, :, 0] # Take only the first code since the main head, will generate it
        motion_embedded = self.motion_embedding(base_motion_tokens)
        
        x = self.pos_encoder(motion_embedded + self.vad_embedding(vad_input) + 
                             self.emotion_embedding(emotion_input) + 
                             self.personality_proj(personality).unsqueeze(1).repeat(1, T, 1) + 
                             self.social_embedding(social).unsqueeze(1).repeat(1, T, 1))
        
        features = self.transformer(tgt=x, memory=memory, tgt_mask=self.get_tgt_mask(T, device))
        
        base_logits = self.base_classifier(features)
        residual_outputs = list(self.residual_predictor(features, memory))
        
        zqs = residual_outputs[:self.residual_predictor.n_tokens]
        residual_logits = residual_outputs[self.residual_predictor.n_tokens:]
        
        logits = torch.stack([base_logits] + residual_logits, dim=-1)
        return logits, zqs