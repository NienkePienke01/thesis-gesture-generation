import torch
import torch.nn as nn
from torch.nn import functional as F

from .rotary_embedding_torch import RotaryEmbedding
from .transformer_modules import FiLMTransformerDecoderLayer, generate_causal_mask

class TurnTakingTransformer(nn.Module):
    def __init__(self, vocab_size=1024, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, 
                 dropout=0.1, downsample_factor=4):
        super().__init__()
        
        self.downsample_factor = downsample_factor
        
        # Parallel input embedding layers for each residual vector quantization codebook layer
        self.motion_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(downsample_factor)
        ])
        
        self.vad_embedding = nn.Embedding(3, d_model) 
        self.emotion_embedding = nn.Embedding(8, d_model) 
        self.social_embedding = nn.Embedding(11, d_model)  
        self.personality_projection = nn.Linear(5, d_model)      

        self.audio_projection = nn.Linear(768, d_model)         
        self.partner_audio_projection = nn.Linear(768, d_model) 
        self.text_projection = nn.Linear(768, d_model)          
        
        self.rotary = RotaryEmbedding(dim=d_model)
        
        self.layers = nn.ModuleList([
            FiLMTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
                rotary=self.rotary
            ) for _ in range(num_layers)
        ])
        
        self.global_conditioning_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Parallel output prediction heads tracking each codebook index slot
        self.motion_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(downsample_factor)
        ])
        self.state_head = nn.Linear(d_model, 3) 

    def forward(self, past_motion_codes, vad_integers, emotion_categories, social_relationship, 
                personality_traits, audio_features, partner_audio_features, bert_features):
        
        batch_size, sequence_length, num_codebooks = past_motion_codes.shape
        device = past_motion_codes.device
        model_hidden_dimension = self.motion_embeddings[0].embedding_dim
        
        # Accumulate parallel multi-codebook input embeddings
        motion_embeddings_sum = torch.zeros((batch_size, sequence_length, model_hidden_dimension), 
                                            device=device, dtype=audio_features.dtype)
        for index in range(num_codebooks):
            motion_embeddings_sum += self.motion_embeddings[index](past_motion_codes[:, :, index])
            
        vad_embeddings = self.vad_embedding(vad_integers)
        emotion_embeddings = self.emotion_embedding(emotion_categories)
        
        social_embeddings = self.social_embedding(social_relationship).unsqueeze(1)
        personality_embeddings = self.personality_projection(personality_traits).unsqueeze(1)
        
        # Merge target stream sequence features element-wise
        main_sequence = motion_embeddings_sum + vad_embeddings + emotion_embeddings + social_embeddings + personality_embeddings
        
        audio_embeddings = self.audio_projection(audio_features)
        partner_audio_embeddings = self.partner_audio_projection(partner_audio_features)
        text_embeddings = self.text_projection(bert_features)
        
        # Compile multi-modal cross-attention memory sequence map
        context_bank = torch.cat([audio_embeddings, partner_audio_embeddings, text_embeddings], dim=1)
        
        # Extract mean-pooled speech vector to drive layer-wise feature scaling
        mean_pooled_audio = audio_embeddings.mean(dim=1)
        global_film_conditioning = self.global_conditioning_projection(mean_pooled_audio)
        
        # Establish causal matrix visibility matching downsampled time axis sequence length
        target_mask = generate_causal_mask(source_length=sequence_length, target_length=sequence_length, device=device)
        
        hidden_state = main_sequence
        for layer in self.layers:
            hidden_state = layer(
                tgt=hidden_state,
                memory=context_bank,
                t=global_film_conditioning,
                tgt_mask=target_mask
            )
        
        # Extract unflattened parallel logit tensors
        predictive_motion_logits = torch.stack([head(hidden_state) for head in self.motion_heads], dim=2)
        predictive_state_logits = self.state_head(hidden_state)
        
        return predictive_motion_logits, predictive_state_logits