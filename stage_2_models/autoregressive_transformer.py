import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """Standard MLP block used in semtalk"""
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
    def forward(self, inputs):
        return self.mlp(inputs)


class PeriodicPositionalEncoding(nn.Module):
    """PeriodicPositionalEncoding used in semtalk"""
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=65): 
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeMask(nn.Module):
    """
    Timemask as used in semtalk
    """
    def __init__(self, max_width: int = 4, num_mask: int = 1):
        super().__init__()
        self.max_width = int(max_width)
        self.num_mask = int(num_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.max_width <= 0 or self.num_mask <= 0:
            return x
        B, T, C = x.shape
        for _ in range(self.num_mask):
            w = int(torch.randint(1, self.max_width + 1, (1,)).item())
            if w >= T:
                continue
            s = int(torch.randint(0, T - w + 1, (1,)).item())
            x[:, s:s+w, :] = 0.0
        return x

class PredictResidualZQ(nn.Module):
    """
    Used to predict the other codes (for example 2-3-4 when you have 4 layers in your vqvae) based on semtalk.py
    """
    def __init__(self, latent_dim=256, num_head=8, ffn_dim=1024, dropout=0.1, n_tokens=3, vocab_size=513):
        super(PredictResidualZQ, self).__init__()
        self.n_tokens = n_tokens
        
        # Cross-attention layers (3 layers each just like semtalk.py)
        cross_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_head, dim_feedforward=ffn_dim, dropout=dropout
        )
        self.cross_attn = nn.ModuleList([
            nn.TransformerDecoder(cross_layer, num_layers=3) for _ in range(n_tokens)
        ])
        
        # Projection maps for concatenated residuals (skipping the first token)
        self.maps = nn.ModuleList([
            nn.Linear(latent_dim * (i + 1), latent_dim) for i in range(1, n_tokens)
        ])
        
        # Classifiers for each level
        self.classifiers = nn.ModuleList([
            MLP(latent_dim, latent_dim, vocab_size) for _ in range(n_tokens)
        ])

    def forward(self, pre_index, cond):
        pre_index = pre_index.permute(1, 0, 2)
        cond = cond.permute(1, 0, 2)
        
        zqs = []
        zq_indices = []
        
        current_input = pre_index
        
        for i in range(self.n_tokens):
            # Apply Cross-Attention
            zq = self.cross_attn[i](tgt=current_input, memory=cond)
            zq = zq + current_input # Residual connection
            
            # Store Result and Classify
            zqs.append(zq)
            zq_indices.append(self.classifiers[i](zq.permute(1, 0, 2)))
            
            # Prepare input for next iteration (if not last)
            if i < self.n_tokens - 1:
                # Concatenate [pre_index, zq_1, ..., zq_i]
                # Each iteration i needs previous zq_1 ... zq_i
                cat_input = torch.cat([pre_index] + zqs, dim=-1)
                current_input = self.maps[i](cat_input)
        
        # permute back to [B, T, C]
        zqs = [zq.permute(1, 0, 2) for zq in zqs]
        
        return *zqs, *zq_indices


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
        
        self.audio_proj = nn.Linear(768, d_model)
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

        
        gating_weights = torch.softmax(self.audio_gate_weights(conversation_state), dim=-1)
        memory = self.memory_fusion(torch.cat([(self.audio_proj(audio) * gating_weights[:, :, 0:1]) + 
                                               (self.audio_proj(partner_audio) * gating_weights[:, :, 1:2]),
                                                 self.bert_proj(bert)
                                                 ], dim=-1))

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