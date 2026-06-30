import argparse
import json
import os
import torch
import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import data_processing.utils as utils
from utils import rotation_conversions as rc
from data_processing.seamless_segmenter import SeamlessSegmenter
from data_processing.seamless_dataset import SeamlessDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def extract_gt_chunks(target_npz_file, segmenter, device, chunk_frames=300):
    """
    Loads a sequence, aligns it using the segmenter, and slices it into
    discrete chunks converted to 6D rotations.
    """
    original_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device, load_vertices=False, with_meta_data=True)
    if original_loader.num_frames <= 0:
        return []

    # Get the aligned data from the segmenter (partner data not needed for pure GT extraction)
    segment_data = segmenter._get_loader_segmentation_data(original_loader, None, None)
    total_frames = segment_data.aligned_pose.shape[0]
    
    # Calculate how many full chunks we can extract
    num_chunks = total_frames // chunk_frames
    extracted_clips = []

    for i in range(num_chunks):
        start_frame = i * chunk_frames 
        end_frame = start_frame + chunk_frames
        
        # Slice the GT pose
        gt_pose_sliced = segment_data.aligned_pose[start_frame:end_frame].unsqueeze(0).to(device)
        
        # Convert Axis-Angle to 6D Rotation matching the exact evaluation format
        batch_size = 1
        full_body_joint_count = 43
        
        gt_pose_matrix = rc.axis_angle_to_matrix(gt_pose_sliced.reshape(batch_size, chunk_frames, full_body_joint_count, 3))
        gt_6d_pose = rc.matrix_to_rotation_6d(gt_pose_matrix)
        
        # Reshape to (chunk_frames, 258)
        gt_6d = gt_6d_pose[0, ...].reshape(chunk_frames, 258).cpu().numpy()
        
        extracted_clips.append({
            "gt": gt_6d
        })

    return extracted_clips

def run_gt_extraction(args):
    device = torch.device(args.device)
    
    print("Initializing Segmenter...")
    with open(args.segmenter_config, 'r') as f:
        seg_config = json.load(f)
    
    # Clean and override segmenter config params
    seg_config = {k: v for k, v in seg_config.items() if "comment" not in k.lower()}
    seg_config["relationship_detail_mandatory"] = False
    seg_config["personality_detail_mandatory"] = False
    seg_config["filter_out_of_screen_hands"] = False
    
    segmenter = SeamlessSegmenter(dataset=None, **seg_config)
    segmenter.bert_model = segmenter.bert_model.to(device)
    segmenter.hubert_model = segmenter.hubert_model.to(device)

    print(f"Loading dataset from {args.data_folder}...")
    dataset = SeamlessDataset(args.data_folder)
    print(f"Found {len(dataset)} valid interaction files based on manifest.")

    # Apply the max_interactions limit if it is provided
    file_ids_to_process = dataset.file_ids
    if args.max_interactions is not None:
        file_ids_to_process = file_ids_to_process[:args.max_interactions]
        print(f"Limiting processing to the first {len(file_ids_to_process)} files.")

    output_dir = PROJECT_ROOT / "evaluation_results" / f"{args.output_name}_chunks_{args.chunk_frames}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(output_dir), map_size=int(1024**3 * args.max_gb))
    sample_idx = 0

    # Pass the sliced list to tqdm
    for file_id in tqdm(file_ids_to_process, desc="Extracting GT Sequences"):
        file_path = dataset.samples.get(file_id)
        if not file_path:
            continue
        
        extracted_clips = extract_gt_chunks(str(file_path), segmenter, device, chunk_frames=args.chunk_frames)
        
        if extracted_clips:
            with env.begin(write=True) as transaction:
                for i, clip in enumerate(extracted_clips):
                    entry = {
                        "gt": clip["gt"],
                        "file_id": f"{file_id}_chunk_{i}"
                    }
                    
                    key = f"{sample_idx:08d}".encode("ascii")
                    transaction.put(key, pickle.dumps(entry))
                    sample_idx += 1

    env.close()
    if sample_idx > 0:
        print(f"Successfully saved {sample_idx} GT chunks ({args.chunk_frames} frames each) to LMDB at {output_dir}")
    else:
        print("No valid sequences were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts Ground Truth data directly to an LMDB for baseline evaluation.")
    parser.add_argument("--segmenter_config", type=str, required=True, help="Path to segmenter configuration")
    parser.add_argument("--data_folder", type=str, required=True, help="Target data folder (train or test)")
    parser.add_argument("--output_name", type=str, default="gt_baseline", help="Prefix name for the output LMDB folder")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_gb", type=int, default=2, help="Max size of LMDB in GB") 
    parser.add_argument("--chunk_frames", type=int, default=300, help="Number of frames per chunk") 
    parser.add_argument("--max_interactions", type=int, default=None, help="Maximum number of interaction files to process.")
    
    run_gt_extraction(parser.parse_args())