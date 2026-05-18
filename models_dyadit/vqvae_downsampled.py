"""
VQ-VAE with Temporal Downsampling
Modified version with 4x or 8x temporal downsampling
"""

import torch
import torch.nn as nn
from vqvae import ResidualVectorQuantization, TemporalVertexCodec


class TemporalVertexEncoderDownsampled(nn.Module):
    """Encoder with temporal downsampling"""
    def __init__(
        self,
        n_vertices: int = 338,
        latent_dim: int = 128,
        downsample_factor: int = 4,  # 2, 4, or 8
    ):
        super().__init__()
        self.input_dim = n_vertices
        self.downsample_factor = downsample_factor
        
        if downsample_factor == 2:
            # 2x downsampling: 64 → 32 tokens
            self.enc = nn.Sequential(
                nn.Conv1d(self.input_dim, latent_dim, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            )
        elif downsample_factor == 4:
            # 4x downsampling: 64 → 16 tokens
            self.enc = nn.Sequential(
                nn.Conv1d(self.input_dim, latent_dim, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            )
        elif downsample_factor == 8:
            # 8x downsampling: 64 → 8 tokens
            self.enc = nn.Sequential(
                nn.Conv1d(self.input_dim, latent_dim, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            raise ValueError(f"Unsupported downsample_factor: {downsample_factor}")

    def forward(self, verts):
        """
        :param verts: B x T x n_vertices tensor
        :return: B x (T // downsample_factor) x latent_dim tensor
        """
        if verts.dim() == 4:
            verts = verts.permute(0, 2, 3, 1).contiguous()
            verts = verts.view(verts.shape[0], self.input_dim, verts.shape[3])
        else:
            verts = verts.permute(0, 2, 1)  # B x C x T
        
        x = self.enc(verts)
        x = x.permute(0, 2, 1).contiguous()  # B x T' x C
        return x

class TemporalVertexDecoderUpsampled(nn.Module):
    """Decoder with temporal upsampling (anti-checkerboard: Upsample + Conv)"""
    def __init__(
        self,
        n_vertices: int = 338,
        latent_dim: int = 128,
        upsample_factor: int = 4,  # 2, 4, or 8
    ):
        super().__init__()
        assert upsample_factor in (2, 4, 8), f"Unsupported upsample_factor: {upsample_factor}"
        self.output_dim = n_vertices
        self.upsample_factor = upsample_factor

        def up_block(c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(c, c, kernel_size=5, padding=2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        blocks = [
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]

        u = upsample_factor
        while u > 1:
            blocks += [
                up_block(latent_dim),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
            u //= 2

        blocks += [nn.Conv1d(latent_dim, self.output_dim, kernel_size=3, padding=1)]
        self.dec = nn.Sequential(*blocks)

    def forward(self, x):
        """
        :param x: B x (T // upsample_factor) x latent_dim tensor
        :return: B x T x n_vertices tensor
        """
        # (B, T', C) -> (B, C, T')
        x = x.permute(0, 2, 1).contiguous()
        verts = self.dec(x)
        # (B, C, T) -> (B, T, C)
        verts = verts.permute(0, 2, 1).contiguous()
        return verts


class TemporalVertexCodecDownsampled(nn.Module):
    """VQ-VAE with temporal downsampling"""
    def __init__(
        self,
        n_vertices: int = 338,
        latent_dim: int = 128,
        categories: int = 128,
        residual_depth: int = 4,
        downsample_factor: int = 4,  # 2, 4, or 8
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.categories = categories
        self.residual_depth = residual_depth
        self.n_clusters = categories
        self.downsample_factor = downsample_factor
        
        self.encoder = TemporalVertexEncoderDownsampled(
            n_vertices=n_vertices, 
            latent_dim=latent_dim,
            downsample_factor=downsample_factor
        )
        self.decoder = TemporalVertexDecoderUpsampled(
            n_vertices=n_vertices, 
            latent_dim=latent_dim,
            upsample_factor=downsample_factor
        )
        self.quantizer = ResidualVectorQuantization(
            dim=latent_dim,
            codebook_size=categories,
            num_quantizers=residual_depth,
            decay=0.99,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
            # commitment_weight uses default 1.0, actual weight controlled in train loop
        )

    def predict(self, verts):
        return self.encode(verts)

    def encode(self, verts):
        """
        :param verts: B x T x n_vertices tensor
        :return: B x (T // downsample_factor) x residual_depth LongTensor
        """
        enc = self.encoder(verts)
        q = self.quantizer.encode(enc)
        return q

    def decode(self, q):
        """
        :param q: B x (T // downsample_factor) x residual_depth LongTensor
        :return: B x T x n_vertices tensor
        """
        reformat = q.dim() > 2
        if reformat:
            B, T_down, _ = q.shape
            q = q.reshape((-1, self.residual_depth))
        enc = self.quantizer.decode(q)
        if reformat:
            enc = enc.reshape((B, T_down, -1))
        verts = self.decoder(enc)
        return verts

    def decode_motion(self, z):
        """
        Decode either discrete codes or continuous latent features back to motion space.
        Accepts:
            - Long / Int tensors of shape [B, T_down, residual_depth] (discrete codes)
            - Float tensors of shape [B, T_down, latent_dim] (continuous latent features)
        Returns:
            Tensor of shape [B, T, n_vertices]
        """
        if z is None:
            raise ValueError("Input to decode_motion cannot be None.")

        if z.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return self.decode(z)

        if not torch.is_floating_point(z):
            raise TypeError(f"Unsupported dtype for decode_motion: {z.dtype}")

        latent = z
        squeeze_first_dim = False
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
            squeeze_first_dim = True
        elif latent.dim() != 3:
            raise ValueError(f"Expected latent tensor with 2 or 3 dims, got shape {latent.shape}")
        verts = self.decoder(latent)
        if squeeze_first_dim:
            verts = verts.squeeze(0)
        return verts

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(
            self.categories, code_idx.shape[0], device=code_idx.device
        )
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def forward(self, verts, mask=None):
        """
        :param verts: B x T x n_vertices tensor
        :return: verts: B x T x n_vertices reconstructed tensor
                 vq_loss: commitment loss
                 perplexity: codebook perplexity
        """
        B, T = verts.shape[0], verts.shape[1]
        x = self.encoder(verts)  # B x (T // downsample_factor) x latent_dim
        T_down = x.shape[1]
        x, code_idx, vq_loss = self.quantizer(
            x.view(B * T_down, self.latent_dim), B, T_down, mask
        )
        perplexity = self.compute_perplexity(code_idx[:, -1].view((-1)))
        verts = self.decoder(x.view(B, T_down, self.latent_dim))
        verts = verts.reshape((verts.shape[0], verts.shape[1], -1))        
        return verts, vq_loss, perplexity