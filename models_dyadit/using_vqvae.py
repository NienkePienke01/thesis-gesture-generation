"""
Minimal example: build the motion VQ-VAE from config, optionally load a checkpoint,
and run encode / decode / codebook lookup.

Layout (flat, all files next to this script):
  ./using_vqvae.py
  ./vqvae.py
  ./vqvae_downsampled.py
  ./model_config.yaml

Run from inside this directory:
  python using_vqvae.py                          # build only, no checkpoint
  python using_vqvae.py /path/to/net_best.pth    # build + load weights
"""

import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import models_dyadit.vqvae_downsampled as vqvae  # noqa: E402


def build_model(config_path: str = None, device: str = None) -> vqvae.TemporalVertexCodecDownsampled:
    """Instantiate the VQ-VAE from a YAML config."""
    if config_path is None:
        config_path = os.path.join(HERE, "model_config.yaml")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path) as f:
        m = yaml.safe_load(f)["model"]

    model = vqvae.TemporalVertexCodecDownsampled(
        n_vertices=m["nb_joints"],
        latent_dim=m["output_emb_width"],
        categories=m["code_dim"],
        residual_depth=m["depth"],
        downsample_factor=m["downsample_factor"],
    ).to(device)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str = None) -> None:
    if device is None:
        device = next(model.parameters()).device
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["net"] if isinstance(state, dict) and "net" in state else state)


# -- convenience wrappers ----------------------------------------------------

def encode_motion(model, motion: torch.Tensor) -> torch.Tensor:
    """motion [B, T, 258] -> tokens [B, T // downsample_factor, depth]."""
    return model.encode(motion)


def decode_motion(model, encoded: torch.Tensor) -> torch.Tensor:
    """tokens [B, T // downsample_factor, depth] -> motion [B, T, 258]."""
    return model.decode(encoded)


def get_codebook_representation(model, encoded: torch.Tensor) -> torch.Tensor:
    """tokens -> continuous codebook embedding [B, T // downsample_factor, latent_dim]."""
    reformat = encoded.dim() > 2
    if reformat:
        B, T_down, _ = encoded.shape
        encoded = encoded.reshape((-1, model.residual_depth))
    rep = model.quantizer.decode(encoded)
    if reformat:
        rep = rep.reshape((B, T_down, -1))
    return rep


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device=device)
    if len(sys.argv) > 1:
        load_checkpoint(model, sys.argv[1])
        print(f"loaded checkpoint: {sys.argv[1]}")
    else:
        print("[note] no checkpoint provided — outputs come from random-init weights")
    model.eval()

    dummy = torch.randn(1, 96, 258, device=device)
    enc = encode_motion(model, dummy)
    print("encoded tokens     :", enc.shape)               # [1, 24, 4]
    print("decoded motion     :", decode_motion(model, enc).shape)  # [1, 96, 258]
    print("codebook embedding :", get_codebook_representation(model, enc).shape)  # [1, 24, 64]
