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


class TurnTakingTransformer(nn.Module):
    def __init__(self, vocab_size=1024, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, 
                 dropout=0.1, max_seq_len=64, activation="gelu"):
        super().__init__()
        # Motion embedding layer used to process all 4 layers
        self.motion_embedding = nn.Embedding(vocab_size, d_model)
        self.vad_embedding = nn.Embedding(3, d_model) 
        self.emotion_embedding = nn.Embedding(8, d_model) 
        
        self.social_embedding = nn.Embedding(11, d_model)  
        self.personality_proj = nn.Linear(5, d_model)      

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout) 
        
        self.audio_proj = nn.Linear(768, d_model)         
        self.partner_audio_proj = nn.Linear(768, d_model) 
        self.text_proj = nn.Linear(768, d_model)          
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Parallel Head Projection Output channels = vocab_size * 4 layers
        self.final_head = nn.Linear(d_model, vocab_size * 4) 
        self.state_head = nn.Linear(d_model, 3) 

    def _generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1) == 1
        return mask.float().masked_fill(mask, float('-inf'))

    def forward(self, past_motion_codes, vad_integers, emotion_categories, social_relationship, 
                personality_traits, audio_features, partner_audio_features, bert_features):
        
        # Expected input shape for past_motion_codes (B, T, 4)
        batch_size, seq_len, num_layers = past_motion_codes.shape
        
        # Embed all 4 code layers -> (B, T, 4, d_model) and sum them to form one frame vector -> (B, T, d_model)
        m_emb = self.motion_embedding(past_motion_codes).sum(dim=2)
        
        v_emb = self.vad_embedding(vad_integers)
        e_emb = self.emotion_embedding(emotion_categories)
        
        soc_emb = self.social_embedding(social_relationship).unsqueeze(1)
        per_emb = self.personality_proj(personality_traits).unsqueeze(1)
        
        main_sequence = m_emb + v_emb + e_emb + soc_emb + per_emb
        main_sequence = self.pos_encoder(main_sequence)
        
        aud_emb = self.audio_proj(audio_features)
        part_aud_emb = self.partner_audio_proj(partner_audio_features)
        txt_emb = self.text_proj(bert_features)
        
        # context_bank = torch.cat([aud_emb, part_aud_emb, txt_emb], dim=1) # Depends if we want to look into the future for context
        context_bank = aud_emb + part_aud_emb + txt_emb # with this we do not
        
        tgt_mask = self._generate_causal_mask(seq_len, past_motion_codes.device)
        
        fused_hidden = self.transformer_decoder(
            tgt=main_sequence,
            memory=context_bank,
            tgt_mask=tgt_mask,
            memory_mask=tgt_mask
        )
        
        # Compute multi-layer classification logits
        logits = self.final_head(fused_hidden) # (B, T, vocab_size * 4)
        pred_logits = logits.reshape(batch_size, seq_len, 4, -1) # Reshaped to (B, T, 4, vocab_size)
        
        state_logits = self.state_head(fused_hidden)
        return pred_logits, state_logits