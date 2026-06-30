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

def calculate_gt_baselines(env, is_warm_start):
    all_gt_static, all_gt_kine = [], []
    seq_mu_s, seq_cov_s, seq_mu_k, seq_cov_k = [], [], [], []

    print("\n" + "="*40)
    print("0. CALCULATING GROUND TRUTH BASELINES")
    print("="*40)

    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Extracting GT"):
            entry = pickle.loads(value)
            gt = convert_6d_to_3d(entry["gt"])
            if is_warm_start:
                gt = gt[64:]
                
            dim = gt.shape[-1]
            gt_s = gt.reshape(-1, dim)
            gt_k = (gt[1:] - gt[:-1]).reshape(-1, dim)
            
            all_gt_static.append(gt_s)
            all_gt_kine.append(gt_k)
            
            m_s, c_s = evaluation.utils.calculate_activation_statistics_np(gt_s)
            m_k, c_k = evaluation.utils.calculate_activation_statistics_np(gt_k)
            seq_mu_s.append((m_s, c_s))
            seq_mu_k.append((m_k, c_k))

    indices = list(range(len(seq_mu_s)))
    rand_idx = indices.copy()
    while any(i == j for i, j in zip(indices, rand_idx)): 
        random.shuffle(rand_idx)
    
    seq_fd_s = [evaluation.utils.calculate_frechet_distance_np(seq_mu_s[i][0], seq_mu_s[i][1], seq_mu_s[j][0], seq_mu_s[j][1]) for i, j in zip(indices, rand_idx)]
    seq_fd_k = [evaluation.utils.calculate_frechet_distance_np(seq_mu_k[i][0], seq_mu_k[i][1], seq_mu_k[j][0], seq_mu_k[j][1]) for i, j in zip(indices, rand_idx)]

    pool_s = np.vstack(all_gt_static)
    pool_k = np.vstack(all_gt_kine)
    
    idx_s = np.random.permutation(len(pool_s))
    idx_k = np.random.permutation(len(pool_k))
    
    pm1_s, pc1_s = evaluation.utils.calculate_activation_statistics_np(pool_s[idx_s[:len(idx_s)//2]])
    pm2_s, pc2_s = evaluation.utils.calculate_activation_statistics_np(pool_s[idx_s[len(idx_s)//2:]])
    pool_fd_s = evaluation.utils.calculate_frechet_distance_np(pm1_s, pc1_s, pm2_s, pc2_s)

    pm1_k, pc1_k = evaluation.utils.calculate_activation_statistics_np(pool_k[idx_k[:len(idx_k)//2]])
    pm2_k, pc2_k = evaluation.utils.calculate_activation_statistics_np(pool_k[idx_k[len(idx_k)//2:]])
    pool_fd_k = evaluation.utils.calculate_frechet_distance_np(pm1_k, pc1_k, pm2_k, pc2_k)

    print(f"DyaDiT (Seq-Avg) -> GT Random FD (Static): {np.mean(seq_fd_s):.4f} | (Kinematic): {np.mean(seq_fd_k):.4f}")
    print(f"DyaPlex (Pooled) -> GT Random FD (Static) [x10^-3]: {pool_fd_s * 1000:.4f} | (Kinematic) [x10^-3]: {pool_fd_k * 1000:.4f}")

    return np.mean(seq_fd_s), np.mean(seq_fd_k), pool_fd_s, pool_fd_k

def main(args):
    env = lmdb.open(args.results, readonly=True)
    is_warm_start = "warm_start_True" in args.results
    print(f"Detected warm_start parameter: {is_warm_start}")
    
    seq_s_fd, seq_k_fd, pool_s_fd, pool_k_fd = calculate_gt_baselines(env, is_warm_start) 
    
    print("\n" + "="*40)
    print("1. EVALUATING GENERATED CLIPS")
    print("="*40)
    
    all_pred_s, all_gt_s = [], []
    all_pred_k, all_gt_k = [], []
    seq_pred_fd_s, seq_pred_fd_k = [], []
    seq_div_s, seq_div_k = [], []
    
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Evaluating"):
            entry = pickle.loads(value)
            
            pred = convert_6d_to_3d(entry["motion"])
            gt = convert_6d_to_3d(entry["gt"])
            
            if is_warm_start:
                pred = pred[64:]
                gt = gt[64:]

            dim = pred.shape[-1]
            pred_s, gt_s = pred.reshape(-1, dim), gt.reshape(-1, dim)
            pred_k, gt_k = (pred[1:] - pred[:-1]).reshape(-1, dim), (gt[1:] - gt[:-1]).reshape(-1, dim)
            
            all_pred_s.append(pred_s); all_gt_s.append(gt_s)
            all_pred_k.append(pred_k); all_gt_k.append(gt_k)
            
            # Seq-Averaged Metrics (DyaDiT style)
            mp_s, cp_s = evaluation.utils.calculate_activation_statistics_np(pred_s)
            mg_s, cg_s = evaluation.utils.calculate_activation_statistics_np(gt_s)
            seq_pred_fd_s.append(evaluation.utils.calculate_frechet_distance_np(mp_s, cp_s, mg_s, cg_s))
            
            mp_k, cp_k = evaluation.utils.calculate_activation_statistics_np(pred_k)
            mg_k, cg_k = evaluation.utils.calculate_activation_statistics_np(gt_k)
            seq_pred_fd_k.append(evaluation.utils.calculate_frechet_distance_np(mp_k, cp_k, mg_k, cg_k))
            
            if pred_s.shape[0] > 1:
                seq_div_s.append(evaluation.utils.calculate_diversity_np(pred_s, min(10000, pred_s.shape[0]-1)))
                seq_div_k.append(evaluation.utils.calculate_diversity_np(pred_k, min(10000, pred_k.shape[0]-1)))
            
    env.close()

    # Pooled Metrics (DyaPlex style)
    p_pred_s, p_gt_s = np.vstack(all_pred_s), np.vstack(all_gt_s)
    p_pred_k, p_gt_k = np.vstack(all_pred_k), np.vstack(all_gt_k)

    m1_s, c1_s = evaluation.utils.calculate_activation_statistics_np(p_pred_s)
    m2_s, c2_s = evaluation.utils.calculate_activation_statistics_np(p_gt_s)
    pool_pred_fd_s = evaluation.utils.calculate_frechet_distance_np(m1_s, c1_s, m2_s, c2_s)

    m1_k, c1_k = evaluation.utils.calculate_activation_statistics_np(p_pred_k)
    m2_k, c2_k = evaluation.utils.calculate_activation_statistics_np(p_gt_k)
    pool_pred_fd_k = evaluation.utils.calculate_frechet_distance_np(m1_k, c1_k, m2_k, c2_k)

    pool_div_s = evaluation.utils.calculate_diversity_np(p_pred_s, min(10000, p_pred_s.shape[0]))
    pool_div_k = evaluation.utils.calculate_diversity_np(p_pred_k, min(10000, p_pred_k.shape[0]))
    
    print("\n" + "="*40)
    print("2. FINAL EVALUATION METRICS")
    print("="*40)
    
    print("--- DyaDiT Style (Sequence Averaged) ---")
    print(f"Static Diversity: {np.mean(seq_div_s):.4f} | Kinematic Diversity: {np.mean(seq_div_k):.4f}")
    print(f"Static FD: {np.mean(seq_pred_fd_s):.4f} | Kinematic FD: {np.mean(seq_pred_fd_k):.4f}")
    
    print("\n--- DyaPlex Style (Pooled) [x10^-3 Scale] ---")
    print(f"Static Diversity: {pool_div_s:.4f} | Kinematic Diversity: {pool_div_k:.4f}")
    print(f"Static FD: {pool_pred_fd_s * 1000:.4f} | Kinematic FD: {pool_pred_fd_k * 1000:.4f}")
    
    input_path = Path(args.results)
    project_name = input_path.stem.split('-')[1] if len(input_path.stem.split('-')) > 1 else "unknown"

    metrics = {
        "project_name": project_name,
        "dyadit_static_div": float(np.mean(seq_div_s)),
        "dyadit_kine_div": float(np.mean(seq_div_k)),
        "dyadit_static_fd": float(np.mean(seq_pred_fd_s)),
        "dyadit_kine_fd": float(np.mean(seq_pred_fd_k)),
        "dyaplex_static_div": float(pool_div_s),
        "dyaplex_kine_div": float(pool_div_k),
        "dyaplex_static_fd": float(pool_pred_fd_s * 1000),
        "dyaplex_kine_fd": float(pool_pred_fd_k * 1000)
    }

    output_dir = Path("evaluation_metrics_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"metrics_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    main(args)