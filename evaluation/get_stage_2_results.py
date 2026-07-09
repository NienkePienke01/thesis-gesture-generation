import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import lmdb
import pickle
import copy
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import time

from stage_2_models.autoregressive_transformer import AutoRegressiveMotionTransformer
from training.stage2_trainer_seamless import load_stage_1_checkpoint
from utils import rotation_conversions as rc
import data_processing.utils as utils
from data_processing.seamless_segmenter import SeamlessSegmenter, get_relationship_id
from data_processing.seamless_dataset import SeamlessDataset
import visualisation.reconstruction_tab_whole_body as stage_1_reconstruction
from visualisation.visualise_stage_2 import sample_top_p_k, generate_motion_sequence, setup_reconstruction, _stage2_runtime_cache, load_stage2_and_stage1_pipeline
from datetime import datetime
from evaluation.utils import set_deterministic_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def reconstruct_and_render_for_eval(target_npz_file, device, top_k, top_p, temperature, warm_start, chunk_frames=300):
    # Load the entire sequence
    start_setup = time.time()
    setup_data = setup_reconstruction(target_npz_file, device, load_vertices=False, max_time=None)
    if isinstance(setup_data, tuple) and len(setup_data) == 2:
        return []

    original_loader, full_segment_data, full_partner_segment_data, relationship = setup_data
    if original_loader.num_frames <= 0:
        return []
    
    setup_duration = time.time() - start_setup

    fps = _stage2_runtime_cache["segmenter"].pose_fps
    audio_hz = _stage2_runtime_cache["segmenter"].audio_hz
    audio_sr = _stage2_runtime_cache["segmenter"].audio_sr
    
    stride = _stage2_runtime_cache["stage1_model"].downsample_factor
    ar_chunk_frames = _stage2_runtime_cache["segmenter"].pose_length 
    
    seed_frames = 64 if warm_start else 0 # Overlapping windows if warm_start
    window_frames = chunk_frames + seed_frames # chunk_frames+64 frames total per generation pass
    
    total_frames = full_segment_data.aligned_pose.shape[0]
    
    # Calculate how many full chunk_frames-frame chunks we can extract, considering the +64 frame seed requirement
    num_10s_clips = (total_frames - seed_frames) // chunk_frames
    if num_10s_clips <= 0:
        return []
    
    generated_clips = []
    batched_poses, batched_wav2vec, batched_raw = [], [], []

    # Gather all chunks
    for i in range(num_10s_clips):
        start_frame = i * chunk_frames 
        end_frame = start_frame + window_frames
        audio_start = int((start_frame / fps) * audio_hz)
        audio_end = int((end_frame / fps) * audio_hz)
        raw_start = int((start_frame / fps) * audio_sr)
        raw_end = int((end_frame / fps) * audio_sr)

        batched_poses.append(full_segment_data.aligned_pose[start_frame:end_frame])
        batched_wav2vec.append(full_segment_data.aligned_wav2vec_audio[audio_start:audio_end])
        
        if hasattr(full_segment_data, 'aligned_raw_audio') and full_segment_data.aligned_raw_audio is not None:
            batched_raw.append(full_segment_data.aligned_raw_audio[raw_start:raw_end])
        else:
            batched_raw.append(None)
    
    # Build batched segments
    batched_segment = copy.copy(full_segment_data)
    batched_segment.aligned_pose = torch.stack([p if torch.is_tensor(p) else torch.from_numpy(p) for p in batched_poses])
    batched_segment.aligned_wav2vec_audio = np.stack(batched_wav2vec)
    if hasattr(full_segment_data, 'personality_vector'):
        batched_segment.personality_vector = np.tile(full_segment_data.personality_vector, (num_10s_clips, 1))
    
    batched_partner = None
    if full_partner_segment_data is not None:
        batched_partner = copy.copy(full_partner_segment_data)
        batched_partner_wav2vec = []
        for i in range(num_10s_clips):
            audio_start = int(((i * chunk_frames) / fps) * audio_hz)
            audio_end = audio_start + int((window_frames / fps) * audio_hz)
            batched_partner_wav2vec.append(full_partner_segment_data.aligned_wav2vec_audio[audio_start:audio_end])
        batched_partner.aligned_wav2vec_audio = np.stack(batched_partner_wav2vec)

    # Generate once for all chunks
    start_gen = time.time()
    num_ar_chunks = math.floor((window_frames - ar_chunk_frames) / stride) + 1
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pose_codes = generate_motion_sequence(
            device, batched_segment, batched_partner, 
            ar_chunk_frames, stride, num_ar_chunks, batch_size=num_10s_clips, 
            temperature=temperature, top_p=top_p, top_k=top_k, warm_start=warm_start
        )
    gen_duration = time.time() - start_gen

    # Decode the batched tokens
    downsample_factor = _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_codes = (ar_chunk_frames // downsample_factor)
    dec_stride = _stage2_runtime_cache["segmenter"].stride

    token_chunks = pose_codes.unfold(dimension=1, size=chunk_codes, step=dec_stride // downsample_factor)
    token_chunks = token_chunks.permute(1, 0, 3, 2)
    pose_codes_list = list(token_chunks.unbind(dim=0))

    num_dec_chunks = len(pose_codes_list)
    valid_frames = (num_dec_chunks - 1) * dec_stride + ar_chunk_frames
    full_body_joint_count = 43
    
    final_reconstructed_6d = torch.zeros((num_10s_clips, valid_frames, full_body_joint_count, 6), device=device)
    pre_frames = ar_chunk_frames - dec_stride
    weight_mask = torch.zeros((num_10s_clips, valid_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=pre_frames, device=device).view(1, -1, 1, 1)

    final_6d_pose = stage_1_reconstruction.decode_sequence(
        device, pose_codes_list, ar_chunk_frames, dec_stride, num_dec_chunks, 
        full_body_joint_count, num_10s_clips, pre_frames, 
        final_reconstructed_6d, weight_mask, fade_in, fade_out, model=_stage2_runtime_cache["stage1_model"]
    )

    # Split the decoded batch back out to save them
    gt_poses = batched_segment.aligned_pose[:, :valid_frames].to(device)
    gt_pose_matrix = rc.axis_angle_to_matrix(gt_poses.reshape(num_10s_clips, valid_frames, 43, 3))
    gt_6d_poses = rc.matrix_to_rotation_6d(gt_pose_matrix)

    min_frames = min(final_6d_pose.shape[1], gt_6d_poses.shape[1])
    
    for b in range(num_10s_clips):
        pred_6d = final_6d_pose[b, :min_frames, ...].reshape(min_frames, 258).cpu().numpy()
        gt_6d = gt_6d_poses[b, :min_frames, ...].reshape(min_frames, 258).cpu().numpy()
        
        if warm_start:
            pred_6d = pred_6d[seed_frames:]
            gt_6d = gt_6d[seed_frames:]
            
        pred_6d = pred_6d[:chunk_frames]
        gt_6d = gt_6d[:chunk_frames]
        
        audio_eval_samples = int((chunk_frames / fps) * audio_sr)
        audio_seed_offset = int((seed_frames / fps) * audio_sr) if warm_start else 0
        
        raw_audio = batched_raw[b]
        if torch.is_tensor(raw_audio):
            raw_audio = raw_audio.cpu().numpy()
            
        audio_cropped = raw_audio[audio_seed_offset : audio_seed_offset + audio_eval_samples] if raw_audio is not None else None

        generated_clips.append({
            "motion": pred_6d,
            "gt": gt_6d,
            "audio": audio_cropped
        })

    print(f"\n[PROFILE] File {target_npz_file}:")
    print(f"    -> Data Setup / Disk Search:  {setup_duration:.2f} seconds")
    print(f"    -> AR Transformer Generation: {gen_duration:.2f} seconds")

    return generated_clips

def run_batch_inference(args):
    set_deterministic_seed(args.seed)
    device = torch.device(args.device)

    print("Initializing pipeline...")
    status = load_stage2_and_stage1_pipeline(
        args.stage2_config, 
        args.stage2_checkpoint, 
        args.segmenter_config, 
        args.device
    )

    print(f"Loading dataset from {args.test_folder}...")
    dataset = SeamlessDataset(args.test_folder)
    print(f"Found {len(dataset)} valid interaction files based on manifest.")

    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    project_name = _stage2_runtime_cache.get('project_name', 'unknown')
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "evaluation_results" / f"evaluation_{start_time}_chunks_{args.chunk_frames}_warm_start_{args.warm_start}-{project_name}"

    output_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(output_dir), map_size=int(1024**3 * args.max_gb))
    sample_idx = 0

    file_ids_to_process = dataset.file_ids
    if args.max_interactions is not None:
        file_ids_to_process = file_ids_to_process[:args.max_interactions]
        print(f"Limiting processing to the first {len(file_ids_to_process)} files.")

    for file_id in tqdm(file_ids_to_process, desc="Generating Gestures"):
        file_path = dataset.samples.get(file_id)
        if not file_path:
            print(f"Warning: {file_id} missing from mapped samples, skipping.")
            continue
        
        generated_clips = reconstruct_and_render_for_eval(
                str(file_path), device, args.top_k, args.top_p, args.temperature, args.warm_start, chunk_frames=args.chunk_frames
            )
        
        if generated_clips:
            with env.begin(write=True) as transaction:
                for i, clip in enumerate(generated_clips):
                    entry = {
                        "motion": clip["motion"],
                        "gt": clip["gt"],
                        "audio": clip["audio"],
                        "file_id": f"{file_id}_chunk_{i}"
                    }
                    
                    key = f"{sample_idx:08d}".encode("ascii")
                    transaction.put(key, pickle.dumps(entry))
                    sample_idx += 1
        else:
            print(f"Warning: {file_id} produced no valid generation clips.")

    env.close()
    if sample_idx > 0:
        print(f"Successfully saved {sample_idx} chunks ({args.chunk_frames} frames each) to LMDB at {output_dir}")
    else:
        print("No valid sequences were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_config", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--segmenter_config", type=str, required=True)
    parser.add_argument("--test_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--max_gb", type=int, default=2) 
    parser.add_argument("--chunk_frames", type=int, default=300) 
    parser.add_argument("--max_interactions", type=int, default=None, help="Maximum number of interaction files to process."),
    parser.add_argument("--seed", type=int, default=42)
    
    run_batch_inference(parser.parse_args())