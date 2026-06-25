import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

from stage_2_models.autoregressive_transformer import AutoRegressiveMotionTransformer
from training.stage2_trainer_seamless import load_stage_1_checkpoint
from utils import rotation_conversions as rc
import data_processing.utils as utils
from data_processing.seamless_segmenter import SeamlessSegmenter, get_relationship_id
from data_processing.seamless_dataset import SeamlessDataset
import visualisation.reconstruction_tab_whole_body as stage_1_reconstruction
from visualisation.visualise_stage_2 import sample_top_p_k, generate_motion_sequence, setup_reconstruction, _stage2_runtime_cache, load_stage2_and_stage1_pipeline
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def reconstruct_and_render_for_eval(target_npz_file, device, top_k, top_p, temperature, warm_start, batch_size):
    setup_data = setup_reconstruction(target_npz_file, device)
    if isinstance(setup_data, tuple) and len(setup_data) == 2: # for if the model is not loaded
        return setup_data

    original_loader, segment_data, partner_segment_data, relationship = setup_data
    stride = _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_frames =  _stage2_runtime_cache["segmenter"].pose_length
    num_frames = segment_data.aligned_pose.shape[0]
    num_chunks = math.floor((num_frames - chunk_frames) / stride) + 1

    # Do inference on stage 2
    pose_codes = generate_motion_sequence(
        device, 
        segment_data, partner_segment_data, 
        chunk_frames, stride, num_chunks, batch_size, 
        temperature=temperature, top_p=top_p, top_k=top_k, warm_start=warm_start
    )

    # Turn the large tensor of motion codes into a list of tensors of size "16" to match stage 1 reconstruction
    downsample_factor =  _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_codes = (chunk_frames // downsample_factor)
    stride = _stage2_runtime_cache["segmenter"].stride # stride value for reconstruction

    # Sliding window chunking to keep stride overlap
    token_chunks = pose_codes.unfold(dimension=1, size=chunk_codes, step=stride // downsample_factor)
    token_chunks = token_chunks.permute(1, 0, 3, 2)
    pose_codes_list = list(token_chunks.unbind(dim=0)) # Permute from [batch, num_chunks, chunk_frames, downsample_factor] to a list of [batch, chunk_frames, downsample_factor] tensors

    # Update the main variables to match the proper iterations of the reconstruction loop
    num_chunks = len(pose_codes_list)
    valid_frames = (num_chunks - 1) * stride + chunk_frames
    full_body_joint_count = 43
    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)
    pre_frames = chunk_frames - stride
    weight_mask = torch.zeros((batch_size, valid_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=pre_frames, device=device).view(1, -1, 1, 1)

    final_6d_pose = stage_1_reconstruction.decode_sequence(
        device, pose_codes_list, chunk_frames, stride, num_chunks, 
        full_body_joint_count, batch_size, pre_frames, 
        final_reconstructed_6d, weight_mask, fade_in, fade_out, model=_stage2_runtime_cache["stage1_model"]
    )

    # Convert Ground Truth to 6D
    gt_pose_sliced = segment_data.aligned_pose[:valid_frames].unsqueeze(0).to(device)
    gt_pose_matrix = rc.axis_angle_to_matrix(gt_pose_sliced.reshape(batch_size, valid_frames, 43, 3))
    gt_6d_pose = rc.matrix_to_rotation_6d(gt_pose_matrix)

    # Crop to min frames to ensure perfectly aligned arrays
    min_frames = min(final_6d_pose.shape[1], gt_6d_pose.shape[1])
    
    pred_6d = final_6d_pose[0, :min_frames, ...].reshape(min_frames, 258)
    gt_6d = gt_6d_pose[0, :min_frames, ...].reshape(min_frames, 258)

    
    fps = _stage2_runtime_cache["segmenter"].pose_fps
    audio_sr = _stage2_runtime_cache["segmenter"].audio_sr
    expected_audio_samples = int((min_frames / fps) * audio_sr)
    
    # Handle tensor vs numpy array gracefully
    raw_audio = segment_data.aligned_raw_audio
    if torch.is_tensor(raw_audio):
        raw_audio = raw_audio.cpu().numpy()
        
    audio_cropped = raw_audio[:expected_audio_samples]

    return pred_6d.cpu().numpy(), gt_6d.cpu().numpy(), audio_cropped

def run_batch_inference(args):
    device = torch.device(args.device)
    
    # Load models into the imported global cache using your function
    print("Initializing pipeline...")
    status = load_stage2_and_stage1_pipeline(
        args.stage2_config, 
        args.stage2_checkpoint, 
        args.segmenter_config, 
        args.device
    )
    print(status)

    # Get the test data
    print(f"Loading dataset from {args.test_folder}...")
    dataset = SeamlessDataset(args.test_folder)
    print(f"Found {len(dataset)} valid interaction files based on manifest.")

    # Run Inference Loop
    all_pred = []
    all_gt = []
    all_audio = []
    
    for file_id in tqdm(dataset.file_ids, desc="Generating Gestures"):
        file_path = dataset.samples.get(file_id)
        if not file_path:
            print(f"Warning: {file_id} missing from mapped samples, skipping.")
            continue
        
        # Get the predicted 6d output values
        pred_6d, gt_6d, raw_audio = reconstruct_and_render_for_eval(
                str(file_path), device, args.top_k, args.top_p, args.temperature, args.warm_start, args.batch_size
            )
        if pred_6d is not None:
            all_pred.append(pred_6d)
            all_gt.append(gt_6d)
            all_audio.append(raw_audio)

    # Save flat arrays for eval.py
    if all_pred:
        stacked_pred = np.concatenate(all_pred, axis=0)
        stacked_gt = np.concatenate(all_gt, axis=0)
        stacked_audio = np.concatenate(all_audio, axis=0)
        
        start_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = PROJECT_ROOT / "evaluation_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the file within that directory
        output_file = output_dir / f"evaluation_{start_time}-{_stage2_runtime_cache['project_name']}.npy"
        np.save(output_file, {"motion": stacked_pred, "gt": stacked_gt, "audio": stacked_audio})
        print(f"Saved evaluation bundle (Shape: {stacked_pred.shape}) to {output_file}")

    else:
        print("No valid sequences were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_config", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--segmenter_config", type=str, required=True)
    parser.add_argument("--test_folder", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--warm_start", action="store_true")
    
    run_batch_inference(parser.parse_args())