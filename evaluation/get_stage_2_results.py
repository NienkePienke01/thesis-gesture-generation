import argparse
import os
import math
import numpy as np
import torch
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from utils import rotation_conversions as rc
from data_processing.seamless_dataset import SeamlessDataset
import visualisation.reconstruction_tab_whole_body as stage_1_reconstruction

# Import exact functions to prevent duplication
from visualisation.visualise_stage_2 import (
    generate_motion_sequence, 
    setup_reconstruction, 
    _stage2_runtime_cache, 
    load_stage2_and_stage1_pipeline
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def reconstruct_and_render_for_eval(setup_data, device, top_k, top_p, temperature, warm_start):
    original_loader, segment_data, partner_segment_data, relationship = setup_data
    
    stride = _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_frames = _stage2_runtime_cache["segmenter"].pose_length
    num_frames = segment_data.aligned_pose.shape[0]
    num_chunks = math.floor((num_frames - chunk_frames) / stride) + 1
    batch_size = 1

    # Call the original, imported inference function directly
    pose_codes = generate_motion_sequence(
        device, 
        segment_data, partner_segment_data, 
        chunk_frames, stride, num_chunks, batch_size, 
        temperature=temperature, top_p=top_p, top_k=top_k, warm_start=warm_start
    )

    downsample_factor = _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_codes = (chunk_frames // downsample_factor)
    stride_recon = _stage2_runtime_cache["segmenter"].stride

    token_chunks = pose_codes.unfold(dimension=1, size=chunk_codes, step=stride_recon // downsample_factor)
    token_chunks = token_chunks.permute(1, 0, 3, 2)
    pose_codes_list = list(token_chunks.unbind(dim=0)) 

    num_chunks_actual = len(pose_codes_list)
    valid_frames = (num_chunks_actual - 1) * stride_recon + chunk_frames
    full_body_joint_count = 43
    
    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)
    pre_frames = chunk_frames - stride_recon
    weight_mask = torch.zeros((batch_size, valid_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=pre_frames, device=device).view(1, -1, 1, 1)

    final_6d_pose = stage_1_reconstruction.decode_sequence(
        device, pose_codes_list, chunk_frames, stride_recon, num_chunks_actual, 
        full_body_joint_count, batch_size, pre_frames, 
        final_reconstructed_6d, weight_mask, fade_in, fade_out, model=_stage2_runtime_cache["stage1_model"]
    )

    # Convert Ground Truth to 6D
    gt_pose_sliced = segment_data.aligned_pose[:valid_frames]
    if not torch.is_tensor(gt_pose_sliced):
        gt_pose_sliced = torch.from_numpy(gt_pose_sliced)
    gt_pose_sliced = gt_pose_sliced.unsqueeze(0).to(device)
    
    gt_pose_matrix = rc.axis_angle_to_matrix(gt_pose_sliced.reshape(batch_size, valid_frames, 43, 3))
    gt_6d_pose = rc.matrix_to_rotation_6d(gt_pose_matrix)

    # Crop to min frames to ensure aligned arrays
    min_frames = min(final_6d_pose.shape[1], gt_6d_pose.shape[1])
    
    pred_6d = final_6d_pose[0, :min_frames, ...].reshape(min_frames, 258)
    gt_6d = gt_6d_pose[0, :min_frames, ...].reshape(min_frames, 258)

    fps = _stage2_runtime_cache["segmenter"].pose_fps
    audio_sr = _stage2_runtime_cache["segmenter"].audio_sr
    expected_audio_samples = int((min_frames / fps) * audio_sr)
    
    raw_audio = segment_data.aligned_raw_audio
    if torch.is_tensor(raw_audio):
        raw_audio = raw_audio.cpu().numpy()
        
    audio_cropped = raw_audio[:expected_audio_samples]

    return pred_6d.cpu().numpy(), gt_6d.cpu().numpy(), audio_cropped

def run_batch_inference(args):
    device = torch.device(args.device)
    
    print("Initializing pipeline...")
    status = load_stage2_and_stage1_pipeline(
        args.stage2_config, args.stage2_checkpoint, args.segmenter_config, args.device
    )
    print(status)

    print(f"Loading dataset from {args.test_folder}...")
    dataset = SeamlessDataset(args.test_folder)
    print(f"Found {len(dataset)} valid interaction files based on manifest.")

    # Parallel Data Loading
    setup_cache = []
    print(f"Loading segment data with {args.workers} background workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(setup_reconstruction, str(dataset.samples.get(fid)), device, False): fid for fid in dataset.file_ids if dataset.samples.get(fid)}
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(future_to_id), desc="Extracting Audio & Context"):
            fid = future_to_id[future]
            try:
                res = future.result()
                if isinstance(res, tuple) and len(res) == 4:
                    setup_cache.append((fid, res))
            except Exception as e:
                print(f"Failed to load {fid}: {e}")

    # Inference stage 2 loop
    all_pred, all_gt, all_audio = [], [], []
    for fid, setup_data in tqdm(setup_cache, desc="Generating Gestures Sequentially"):
        try:
            pred_6d, gt_6d, raw_audio = reconstruct_and_render_for_eval(
                setup_data, device, args.top_k, args.top_p, args.temperature, args.warm_start
            )
            if pred_6d is not None:
                all_pred.append(pred_6d)
                all_gt.append(gt_6d)
                all_audio.append(raw_audio)
        except Exception as e:
            print(f"Skipping {fid} due to error during generation: {e}")

    if all_pred:
        stacked_pred = np.concatenate(all_pred, axis=0)
        stacked_gt = np.concatenate(all_gt, axis=0)
        stacked_audio = np.concatenate(all_audio, axis=0)
        
        start_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = PROJECT_ROOT / "evaluation_results"
        output_dir.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--warm_start", action="store_true")
    
    run_batch_inference(parser.parse_args())