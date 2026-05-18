# Motion VQ-VAE — Inference Code Bundle (no weights)

Model definition + minimal loader for a Residual VQ-VAE that compresses
6D-rotation upper-body + hands motion (258 dims, 30 fps) into discrete tokens
with 4× temporal downsampling.

**Weights are not included.** Provide a `net_best.pth` checkpoint at load time.

## Contents
```
.
├── README.md
├── using_vqvae.py          # example: build model + (optional) load ckpt
├── vqvae.py                # base modules (TemporalVertexCodec, ResidualVectorQuantization)
├── vqvae_downsampled.py    # 2x/4x/8x downsampling wrapper
└── model_config.yaml       # architecture hyperparameters
```

## Model spec (`model_config.yaml`)
| key                  | value | meaning                                |
| -------------------- | ----- | -------------------------------------- |
| `nb_joints`          | 258   | input dim (6D rotation × joints)       |
| `output_emb_width`   | 64    | latent dim                             |
| `code_dim`           | 512   | codebook size                          |
| `depth`              | 4     | residual depth (4 codebooks)           |
| `downsample_factor`  | 4     | temporal compression (96 frames → 24)  |

I/O shapes (window of 96 frames @ 30 fps = 3.2 s):
- input motion:    `[B, 96, 258]`
- token IDs:       `[B, 24, 4]` (24 time steps × 4 residual codebooks)
- codebook embed:  `[B, 24, 64]` (continuous, used as conditioning by downstream models)
- decoded motion:  `[B, 96, 258]`

## Requirements
- Python 3.10+
- `torch` (any recent 2.x)
- `einops`
- `pyyaml`

## Usage
```bash
cd vqvae_release/
# Run with random-init weights (smoke test)
python using_vqvae.py

# Run with your trained checkpoint
python using_vqvae.py /path/to/net_best.pth
```

In your own code:
```python
import sys
sys.path.insert(0, "/path/to/vqvae_release")
from using_vqvae import build_model, load_checkpoint
import torch

model = build_model()                          # reads ./model_config.yaml
load_checkpoint(model, "/path/to/net_best.pth")
model.eval()

motion = torch.randn(1, 96, 258, device="cuda")
tokens = model.encode(motion)                  # [1, 24, 4]
recon  = model.decode(tokens)                  # [1, 96, 258]
```
