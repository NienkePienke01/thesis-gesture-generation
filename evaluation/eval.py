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

def calculate_gt_baselines(env, is_warm_start):
    all_gt_static_frames, all_gt_kine_frames = [], []
    clip_static_list, clip_kine_list = [], []
    
    print("\n" + "="*40)
    print("0. CALCULATING GROUND TRUTH BASELINES (POOLED)")
    print("="*40)
    
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Extracting GT Frames"):
            entry = pickle.loads(value)
            gt_3d = evaluation.utils.convert_6d_to_3d(entry["gt"])
                
            feature_dim = gt_3d.shape[-1]
            gt_s = gt_3d.reshape(-1, feature_dim)
            gt_k = (gt_3d[1:, :] - gt_3d[:-1, :]).reshape(-1, feature_dim)
            
            all_gt_static_frames.append(gt_s)
            all_gt_kine_frames.append(gt_k)
            clip_static_list.append(gt_s)
            clip_kine_list.append(gt_k)

    pooled_gt_static = np.vstack(all_gt_static_frames)
    pooled_gt_kine = np.vstack(all_gt_kine_frames)
    
    div_samples_g = min(10000, pooled_gt_static.shape[0])
    div_samples_k = min(10000, pooled_gt_kine.shape[0])
    
    static_gt_div = float(evaluation.utils.calculate_diversity_np(pooled_gt_static, div_samples_g))
    kine_gt_div = float(evaluation.utils.calculate_diversity_np(pooled_gt_kine, div_samples_k))

    num_clips = len(clip_static_list)
    random_indices = np.random.choice(num_clips, size=num_clips, replace=True)

    random_gt_static = np.vstack([clip_static_list[i] for i in random_indices])
    random_gt_kine = np.vstack([clip_kine_list[i] for i in random_indices])

    mu1_s, cov1_s = evaluation.utils.calculate_activation_statistics_np(pooled_gt_static)
    mu2_s, cov2_s = evaluation.utils.calculate_activation_statistics_np(random_gt_static)
    static_gt_random_fd = float(evaluation.utils.calculate_frechet_distance_np(mu1_s, cov1_s, mu2_s, cov2_s))

    mu1_k, cov1_k = evaluation.utils.calculate_activation_statistics_np(pooled_gt_kine)
    mu2_k, cov2_k = evaluation.utils.calculate_activation_statistics_np(random_gt_kine)
    kine_gt_random_fd = float(evaluation.utils.calculate_frechet_distance_np(mu1_k, cov1_k, mu2_k, cov2_k))

    print(f"GT Diversity (Static): {static_gt_div:.4f}")
    print(f"GT Random FD (Static) [x10^3 Scale]: {static_gt_random_fd * 1000:.4f}")
    print(f"GT Diversity (Kinematic): {kine_gt_div:.4f}")
    print(f"GT Random FD (Kinematic) [x10^3 Scale]: {kine_gt_random_fd * 1000:.4f}")
    
    return static_gt_div, static_gt_random_fd, kine_gt_div, kine_gt_random_fd


def main(args):
    env = lmdb.open(args.results, readonly=True)
    is_warm_start = "warm_start_True" in args.results
    print(f"Detected warm_start parameter: {is_warm_start}")
    
    static_gt_div, static_gt_random_fd, kine_gt_div, kine_gt_random_fd = calculate_gt_baselines(env, is_warm_start) 
    
    print("\n" + "="*40)
    print("1. EVALUATING CLIPS FROM LMDB (POOLED)")
    print("="*40)
    
    all_pred_static, all_gt_static = [], []
    all_pred_kine, all_gt_kine = [], []
    
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Extracting Generated Frames"):
            entry = pickle.loads(value)
            
            pred_3d = evaluation.utils.convert_6d_to_3d(entry["motion"])
            gt_3d = evaluation.utils.convert_6d_to_3d(entry["gt"])
            
            if is_warm_start:
                pred_3d = pred_3d[64:]
                gt_3d = gt_3d[64:]

            feature_dim = pred_3d.shape[-1]
            
            all_pred_static.append(pred_3d.reshape(-1, feature_dim))
            all_gt_static.append(gt_3d.reshape(-1, feature_dim))
            
            all_pred_kine.append((pred_3d[1:, :] - pred_3d[:-1, :]).reshape(-1, feature_dim))
            all_gt_kine.append((gt_3d[1:, :] - gt_3d[:-1, :]).reshape(-1, feature_dim))
            
    env.close()

    pooled_pred_static = np.vstack(all_pred_static)
    pooled_gt_static = np.vstack(all_gt_static)
    pooled_pred_kine = np.vstack(all_pred_kine)
    pooled_gt_kine = np.vstack(all_gt_kine)

    div_samples_g = min(10000, pooled_pred_static.shape[0])
    div_samples_k = min(10000, pooled_pred_kine.shape[0])

    static_div = float(evaluation.utils.calculate_diversity_np(pooled_pred_static, div_samples_g))
    kinematic_div = float(evaluation.utils.calculate_diversity_np(pooled_pred_kine, div_samples_k))

    mu_pred_static, cov_pred_static = evaluation.utils.calculate_activation_statistics_np(pooled_pred_static)
    mu_gt_static, cov_gt_static = evaluation.utils.calculate_activation_statistics_np(pooled_gt_static)
    static_fd = float(evaluation.utils.calculate_frechet_distance_np(mu_pred_static, cov_pred_static, mu_gt_static, cov_gt_static))

    mu_pred_kine, cov_pred_kine = evaluation.utils.calculate_activation_statistics_np(pooled_pred_kine)
    mu_gt_kine, cov_gt_kine = evaluation.utils.calculate_activation_statistics_np(pooled_gt_kine)
    kinematic_fd = float(evaluation.utils.calculate_frechet_distance_np(mu_pred_kine, cov_pred_kine, mu_gt_kine, cov_gt_kine))
    
    print("\n" + "="*40)
    print("2. FINAL EVALUATION METRICS")
    print("="*40)
    
    print(f"Static Diversity: {static_div:.4f}")
    # Applying the 1000 multiplier to match the papers' tables
    print(f"Static FD [x10^3 Scale]: {static_fd * 1000:.4f}")
    print(f"Kinematic Diversity: {kinematic_div:.4f}")
    print(f"Kinematic FD [x10^3 Scale]: {kinematic_fd * 1000:.4f}")
    
    input_path = Path(args.results)
    project_name = input_path.stem.split('-')[1] if len(input_path.stem.split('-')) > 1 else "unknown"

    metrics = {
        "project_name": project_name,
        "static_diversity": static_div,
        "kinematic_diversity": kinematic_div, 
        "static_FD": static_fd * 1000,
        "kinematic_FD": kinematic_fd * 1000,
        "static GT diversity": static_gt_div, 
        "static GT FD": static_gt_random_fd * 1000, 
        "Kinematic GT diversity": kine_gt_div,
        "Kinematic GT FD": kine_gt_random_fd * 1000
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