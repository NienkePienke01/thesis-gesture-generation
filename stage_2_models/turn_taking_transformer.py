import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CoarsePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1, L=4):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        
        # Force the position tensor to group identical indexes together
        position = (torch.arange(0, max_len, dtype=torch.float) // L).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TurnTakingTransformer(nn.Module):
    def __init__(self, vocab_size=513, d_model=256, nhead=8, num_layers=2, dim_feedforward=1024, 
                 dropout=0.1, max_seq_len=64, activation="gelu", downsample_factor=4):
        super().__init__()
        self.residual_depth = 4
        
        # Keep embeddings standard
        self.motion_embedding = nn.Embedding(vocab_size, d_model)
        
        # NEW: A linear layer to merge your L=4 code embeddings into a single frame vector
        self.code_fusion = nn.Linear(self.residual_depth * d_model, d_model)
        
        self.vad_embedding = nn.Embedding(3, d_model) 
        self.emotion_embedding = nn.Embedding(8, d_model) 
        self.social_embedding = nn.Embedding(11, d_model)  
        self.personality_proj = nn.Linear(5, d_model)      

        # Standard positional encoding (No more // L hacking needed!)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout) 
        
        self.audio_proj = nn.Linear(768, d_model)         
        self.partner_audio_proj = nn.Linear(768, d_model) 
        self.text_proj = nn.Linear(768, d_model)          
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, activation=activation, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Hierarchical output heads: One head per codebook layer
        self.motion_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(self.residual_depth)
        ])
        self.state_head = nn.Linear(d_model, 3) 

    def forward(self, past_motion_codes, vad_integers, emotion_categories, social_relationship, 
                personality_traits, audio_features, partner_audio_features, bert_features):
        
        # past_motion_codes shape: (B, T, L)
        batch_size, seq_len, _ = past_motion_codes.shape
        
        # Embed all codes: (B, T, L, d_model)
        m_emb = self.motion_embedding(past_motion_codes) 
        # Flatten the codes per frame: (B, T, L * d_model)
        m_emb = m_emb.view(batch_size, seq_len, self.residual_depth * -1)
        # Fuse into a single frame embedding: (B, T, d_model)
        m_emb = self.code_fusion(m_emb)
        
        v_emb = self.vad_embedding(vad_integers)
        e_emb = self.emotion_embedding(emotion_categories)
        soc_emb = self.social_relationship(social_relationship).unsqueeze(1)
        per_emb = self.personality_proj(personality_traits).unsqueeze(1)
        
        main_sequence = self.pos_encoder(m_emb + v_emb + e_emb + soc_emb + per_emb)
        
        context_bank = torch.cat([
            self.audio_proj(audio_features), 
            self.partner_audio_proj(partner_audio_features), 
            self.text_proj(bert_features)
        ], dim=1) 
        
        # Clean square causal mask matching frame length T
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=past_motion_codes.device)
        
        fused_hidden = self.transformer_decoder(
            tgt=main_sequence, memory=context_bank, tgt_mask=tgt_mask, memory_mask=None
        )
        
        # Project hidden states into independent code streams: (B, T, L, vocab_size)
        pred_logits = torch.stack([head(fused_hidden) for head in self.motion_heads], dim=2)
        state_logits = self.state_head(fused_hidden)
        
        return pred_logits, state_logits