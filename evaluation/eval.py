import os
import librosa
from evaluation.audio_utils import peak_pick
librosa.util.peak_pick = peak_pick
import argparse
from scipy.signal import argrelextrema
import math
from scipy import linalg
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple
import torch
import numpy as np
from data_processing.utils import get_smplh_model
from utils import rotation_conversions as rc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
import lmdb
import pickle
import random
import evaluation.utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- SKELETON CONFIGURATION ---
L_SHOULDER, L_ELBOW, L_WRIST = 16, 18, 20
R_SHOULDER, R_ELBOW, R_WRIST = 17, 19, 21
# ------------------------------
def convert_6d_to_3d(pose_6d: np.ndarray, device='cuda') -> np.ndarray:
    frames = pose_6d.shape[0]
    chunk_size = 512
    
    # Convert to Torch Tensor and Reshape
    pose_6d_tensor = torch.tensor(pose_6d, dtype=torch.float32, device=device).reshape(frames, 43, 6)
    
    # Convert 6D -> Matrix -> Axis-Angle and flatten to (frames, 129)
    pose_axis_angle = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(pose_6d_tensor)).reshape(frames, -1)

    num_frames = pose_axis_angle.shape[0]
    
    # Split the 129-dim vector back into its components
    # Indices: Body (0:39), Left Hand (39:84), Right Hand (84:129)
    rec_body_flat = pose_axis_angle[:, :39]
    rec_left_hand_flat = pose_axis_angle[:, 39:84]
    rec_right_hand_flat = pose_axis_angle[:, 84:129]

    # Reshape components
    rec_body = rec_body_flat.reshape(num_frames, 13, 3)
    left_hand = rec_left_hand_flat.reshape(num_frames, 15, 3)
    right_hand = rec_right_hand_flat.reshape(num_frames, 15, 3)

    # Create a full 21-joint body pose tensor initialized to zero
    full_body_pose = torch.zeros((num_frames, 21, 3), device=device)

    body_mask = [0, 1, 3, 4, 6, 7, 9, 10]
    keep_indices = [i for i in range(21) if i not in body_mask]
    eval_indices = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + list(range(22, 52))
    # eval_indices = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] + list(range(22, 52))
    
    # Map the 13 reconstructed joints back to their original skeletal indices
    full_body_pose[:, keep_indices, :] = rec_body
    smplh = get_smplh_model(device, gender='male')

    all_joints_3d = []

    for i in range(0, num_frames, chunk_size):
        end = min(i + chunk_size, num_frames)
        curr_batch_size = end - i
        smplh.batch_size = curr_batch_size

        # Slice the inputs specifically for the current batch and flatten the last dimension
        current_batch = {
            'body_pose': full_body_pose[i:end].reshape(curr_batch_size, -1),
            'left_hand_pose': left_hand[i:end].reshape(curr_batch_size, -1),
            'right_hand_pose': right_hand[i:end].reshape(curr_batch_size, -1),
            'global_orient': torch.zeros((curr_batch_size, 3), device=device),
            'transl': torch.zeros((curr_batch_size, 3), device=device),
            'betas': smplh.betas.expand(curr_batch_size, -1),
            'rot2pose': True
        }

        with torch.no_grad():
            output = smplh(**current_batch)
            joints_3d = output.joints[:, eval_indices, :]
            all_joints_3d.append(joints_3d.cpu().numpy())
        
    return np.concatenate(all_joints_3d, axis=0).reshape(frames, -1)

def calculate_gt_baselines(env):
    all_gt_static_stats, all_gt_kine_stats = [], []
    all_gt_static_div, all_gt_kine_div = [], []
    
    print("\n" + "="*40)
    print("0. CALCULATING GROUND TRUTH BASELINES")
    print("="*40)
    
    # Gather all GT statistics and per-sequence diversities
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Extracting GT Stats"):
            entry = pickle.loads(value)
            gt_3d = convert_6d_to_3d(entry["gt"])
            feature_dim = gt_3d.shape[-1]
            
            gt_static = gt_3d.reshape(-1, feature_dim)
            gt_kine = (gt_3d[1:, :] - gt_3d[:-1, :]).reshape(-1, feature_dim)

            # Calculate means and covariances
            mu_static, cov_static = evaluation.utils.calculate_activation_statistics_np(gt_static)
            mu_kine, cov_kine = evaluation.utils.calculate_activation_statistics_np(gt_kine)
            
            all_gt_static_stats.append((mu_static, cov_static))
            all_gt_kine_stats.append((mu_kine, cov_kine))

            # Calculate and store diversity metrics
            div_samples_g = min(10000, gt_static.shape[0] - 1)
            div_samples_k = min(10000, gt_kine.shape[0] - 1)

            if div_samples_g > 0:
                all_gt_static_div.append(evaluation.utils.calculate_diversity_np(gt_static, div_samples_g))
            if div_samples_k > 0:
                all_gt_kine_div.append(evaluation.utils.calculate_diversity_np(gt_kine, div_samples_k))

    num_clips = len(all_gt_static_stats)
    if num_clips < 2:
        return None

    # Create mismatched pairs for cross-sample FD
    indices = list(range(num_clips))
    random_indices = indices.copy()
    while True:
        random.shuffle(random_indices)
        if not any(i == j for i, j in zip(indices, random_indices)):
            break

    # Calculate FD between the random pairs
    gt_random_static_fd, gt_random_kine_fd = [], []
    for i, j in zip(indices, random_indices):
        # Static FD
        mu1_g, cov1_g = all_gt_static_stats[i]
        mu2_g, cov2_g = all_gt_static_stats[j]
        gt_random_static_fd.append(evaluation.utils.calculate_frechet_distance_np(mu1_g, cov1_g, mu2_g, cov2_g))
        
        # Kinematic FD
        mu1_k, cov1_k = all_gt_kine_stats[i]
        mu2_k, cov2_k = all_gt_kine_stats[j]
        gt_random_kine_fd.append(evaluation.utils.calculate_frechet_distance_np(mu1_k, cov1_k, mu2_k, cov2_k))
        
    static_gt_div = float(np.mean(all_gt_static_div))
    kine_gt_div = float(np.mean(all_gt_kine_div))
    static_gt_random_fd = float(np.mean(gt_random_static_fd))
    kine_gt_random_fd = float(np.mean(gt_random_kine_fd))
    
    print(f"GT Diversity (Static): {static_gt_div:.4f}")
    print(f"GT Random FD (Static): {static_gt_random_fd:.4f}")
    print(f"GT Diversity (Kinematic): {kine_gt_div:.4f}")
    print(f"GT Random FD (Kinematic): {kine_gt_random_fd:.4f}")
    
    return static_gt_div, static_gt_random_fd, kine_gt_div, kine_gt_random_fd


def main(args):
    env = lmdb.open(args.results, readonly=True)
    is_warm_start = "warm_start_True" in args.results
    print(f"Detected warm_start parameter: {is_warm_start}")
    if is_warm_start:
        print("Note: Stripping the first 64 seed frames from all clips before evaluation.")
    
    static_gt_div, static_gt_random_fd, kine_gt_div, kine_gt_random_fd = calculate_gt_baselines(env) 
    
    print("="*40)
    print("1. EVALUATING CLIPS FROM LMDB (Per-Sequence)")
    print("="*40)
    
    all_static_div, all_kinematic_div = [], []
    all_static_fd, all_kinematic_fd = [], []
    
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Evaluating Clips"):
            entry = pickle.loads(value)
            
            # Convert 6D to 3D for this specific clip
            pred_3d = convert_6d_to_3d(entry["motion"])
            gt_3d = convert_6d_to_3d(entry["gt"])
            
            if is_warm_start:
                pred_3d = pred_3d[64:]
                gt_3d = gt_3d[64:]

            feature_dim = pred_3d.shape[-1]
            
            # Static Formatting
            pred_static = pred_3d.reshape(-1, feature_dim)
            gt_static = gt_3d.reshape(-1, feature_dim)
            
            # Kinematic Formatting (Velocity)
            pred_kine = (pred_3d[1:, :] - pred_3d[:-1, :]).reshape(-1, feature_dim)
            gt_kine = (gt_3d[1:, :] - gt_3d[:-1, :]).reshape(-1, feature_dim)
            
            # Ensure diversity_times is strictly smaller than the number of frames in the clip
            div_samples_g = min(10000, pred_static.shape[0] - 1)
            div_samples_k = min(10000, pred_kine.shape[0] - 1)
            
            # Per-sequence Diversity
            if div_samples_g > 0:
                all_static_div.append(evaluation.utils.calculate_diversity_np(pred_static, div_samples_g))
            if div_samples_k > 0:
                all_kinematic_div.append(evaluation.utils.calculate_diversity_np(pred_kine, div_samples_k))
            
            # Per-sequence FD statistics calculations
            mu_pred_static, cov_pred_static = evaluation.utils.calculate_activation_statistics_np(pred_static)
            mu_gt_static, cov_gt_static = evaluation.utils.calculate_activation_statistics_np(gt_static)
            
            mu_pred_kine, cov_pred_kine = evaluation.utils.calculate_activation_statistics_np(pred_kine)
            mu_gt_kine, cov_gt_kine = evaluation.utils.calculate_activation_statistics_np(gt_kine)

            # Per-sequence FD
            all_static_fd.append(evaluation.utils.calculate_frechet_distance_np(mu_pred_static, cov_pred_static, mu_gt_static, cov_gt_static))
            all_kinematic_fd.append(evaluation.utils.calculate_frechet_distance_np(mu_pred_kine, cov_pred_kine, mu_gt_kine, cov_gt_kine))
            
    env.close()

    # Average everything together
    static_div = float(np.mean(all_static_div))
    kinematic_div = float(np.mean(all_kinematic_div))
    static_fd = float(np.mean(all_static_fd))
    kinematic_fd = float(np.mean(all_kinematic_fd))
    
    print("\n" + "="*40)
    print("2. FINAL EVALUATION METRICS (Averaged)")
    print("="*40)
    
    print(f"Static Diversity: {static_div:.4f}")
    print(f"Static FD: {static_fd:.4f}")
    print(f"Kinematic Diversity: {kinematic_div:.4f}")
    print(f"Kinematic FD: {kinematic_fd:.4f}")
    
    # Save the metrics
    input_path = Path(args.results)
    project_name = input_path.stem.split('-')[1] if len(input_path.stem.split('-')) > 1 else "unknown"

    metrics = {
        "project_name": project_name,
        "static_diversity": static_div,
        "kinematic_diversity": kinematic_div, 
        "static_FD": static_fd,
        "kinematic_FD": kinematic_fd,
        "static GT diversity": static_gt_div, 
        "static GT FD": static_gt_random_fd, 
        "Kinematic GT diversity": kine_gt_div,
        "Kinematic GT FD": kine_gt_random_fd
    }

    output_dir = Path("evaluation_metrics_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"metrics_{project_name}_{start_time}.json"
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved evaluation metrics for '{project_name}' to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    main(args)