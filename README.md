# Seamless Interaction Project
---

## Project Structure

* **cache/**: Contains generated LMDB datasets and CSV filtering metadata.
* **checkpoints/**: Stores the best model weights (`.bin` files).
* **configs/**: JSON files for data processing and model training.
* **data_processing/**: Scripts for segmenting raw motion data and loading LMDB records.
* **models/**: Contains the architecture (from SemTalk)
* **smpl_models/**: Directory for SMPL-H and SMPL-X body models.
* **training/**: Contains the trainer architecture and the script to train the first stage.
* **visualisation/**: Contains the visaluser to visualise .npz files and .mp4 files from beat and seamless interaction
* **utils/**: Contains utils scripts (from SemTalk)

---

## Configuration Types

# Configuration Details

The project utilizes two primary JSON configuration files to manage the data pipeline and the neural network architecture. Below is a detailed technical breakdown of each parameter.

---

## Config Types

### Config for the segmenter

* **pose_fps (30)**: How many frames per second the pose should be sampled in
* **audio_sr (16000)**: The target sample rate for audio
* **pose_length (64)**: The number of frames per training sample
* **stride (64)**: The step size between segments. Setting this to equal to pose_length ensures no overlap between clips
* **max_gb**: the maximum storage limit for the LMDB environment
* **clean_first_seconds**: Can be used to trim off the first seconds (to prevent t-pose data)
* **clean_final_seconds**: Can be used to trim off the final seconds (to prevent t-pose data)

---

### Config for the training of stage 1

#### general
* **vae_test_dim (78)**: The input feature dimension. In `upper_body` mode, this corresponds to **13 joints** represented in **6D rotation** format ($13 \times 6 = 78$).
* **mode**: `upper_body` or `hands` or `full` decides what part this model will be training for
* **device**: `cuda` or `cpu`
* **gpus**: list of gpus to use during training such as `[0]`

#### data
* **train_lmdb**: the path to the training lmdb, relative to the project root
* **val_lmdb**: the path to the validation lmdb, relative to the project root

#### model_params for the VAE
* **nb_code**: The codebook size
* **code_dim**: The dimensionality of the latent embeddings stored in the codebook
* **output_emb_width**:
* **down_t**: the temporal compression factor.
* **depth**: the number of residual blocks within each ResNet layer
* **stride_t**
* **width**
* **depth**
* **dilation_growth_rate**
* **activation**

#### training_params
* **lr_base**: The starting learning rate for the Adam optimizer.
* **grad_norm**: The threshold for **gradient clipping**. This prevents "exploding gradients" which can occur during the complex kinematic calculations of the vertex loss.
* **batch_size**
* **epochs**
* **decay_epochs**

#### loss_weights
* **rec_weight**: The multiplier for reconstruction-related losses.
* **rec_pos_weight**: The specific weight for **Geodesic Loss**
* **rec_ver_weight**: The weight for the **Vertex Loss**
---

## Setup and Installation

### Dependencies
This project uses `uv` to install the environment use
```bash
uv sync
```

### Follow these commands in to prepare data and begin training:
* **Download specific subsets of the dataset using filters.** `uv run python -m seamless_downloading.seamless_downloader --download_name validation_testing --filter_dict '{"label": "naturalistic", "split": "dev"}' --preferred_vendors True --force_pairs True --num_samples 2 --random_draw True --seed 42`
* **Convert raw files into LMDB format for training access** `uv run -m data_processing.seamless_segmenter --dataset_path data/seamless/downloads/validation_testing --config configs/seamless_segmenter.json`
* **Launch the Stage 1 training process.** `uv run -m training.train_ae_trainer_seamless --config configs/seamless_stage_1_upper_training.json`
* **Run the Gradio interface to inspect/compare motions and videos** `uv run python -m visualisation.gradio_visualisation`
