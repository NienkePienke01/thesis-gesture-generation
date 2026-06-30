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

def main(args):
    env = lmdb.open(args.results, readonly=True)
    
    print("\n" + "="*40)
    print("1. EVALUATING GENERATED CLIPS FROM LMDB")
    print("="*40)
    
    all_pred_s_seqs, all_pred_k_seqs = [], []
    all_gt_s_seqs, all_gt_k_seqs = [], []
    
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Extracting Frames"):
            entry = pickle.loads(value)
            
            pred_3d = evaluation.utils.convert_6d_to_3d(entry["motion"])
            gt_3d = evaluation.utils.convert_6d_to_3d(entry["gt"])

            dim = pred_3d.shape[-1]
            
            all_pred_s_seqs.append(pred_3d.reshape(-1, dim))
            all_gt_s_seqs.append(gt_3d.reshape(-1, dim))
            
            all_pred_k_seqs.append((pred_3d[1:] - pred_3d[:-1]).reshape(-1, dim))
            all_gt_k_seqs.append((gt_3d[1:] - gt_3d[:-1]).reshape(-1, dim))
            
    env.close()

    # Pooled Diversity
    pooled_pred_s = np.vstack(all_pred_s_seqs)
    pooled_pred_k = np.vstack(all_pred_k_seqs)

    div_samples_s = min(10000, pooled_pred_s.shape[0])
    div_samples_k = min(10000, pooled_pred_k.shape[0])

    static_div = float(evaluation.utils.calculate_diversity_np(pooled_pred_s, div_samples_s))
    kinematic_div = float(evaluation.utils.calculate_diversity_np(pooled_pred_k, div_samples_k))

    # Sequence-Averaged FD
    seq_fd_s, seq_fd_k = [], []
    for i in range(len(all_pred_s_seqs)):
        # Calculate FD between generated sequence and its corresponding GT sequence
        mp_s, cp_s = evaluation.utils.calculate_activation_statistics_np(all_pred_s_seqs[i])
        mg_s, cg_s = evaluation.utils.calculate_activation_statistics_np(all_gt_s_seqs[i])
        seq_fd_s.append(evaluation.utils.calculate_frechet_distance_np(mp_s, cp_s, mg_s, cg_s))
        
        mp_k, cp_k = evaluation.utils.calculate_activation_statistics_np(all_pred_k_seqs[i])
        mg_k, cg_k = evaluation.utils.calculate_activation_statistics_np(all_gt_k_seqs[i])
        seq_fd_k.append(evaluation.utils.calculate_frechet_distance_np(mp_k, cp_k, mg_k, cg_k))

    static_fd = float(np.mean(seq_fd_s))
    kinematic_fd = float(np.mean(seq_fd_k))
    
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(f"Static Diversity: {static_div:.4f}")
    print(f"Kinematic Diversity: {kinematic_div:.4f}")
    print(f"Static FD:  {static_fd:.4f}")
    print(f"Kinematic FD: {kinematic_fd:.4f}")
    
    input_path = Path(args.results)
    project_name = input_path.stem.split('-')[1] if len(input_path.stem.split('-')) > 1 else "unknown"

    metrics = {
        "project_name": project_name,
        "static_diversity": static_div,
        "kinematic_diversity": kinematic_div, 
        "static_FD": static_fd,
        "kinematic_FD": kinematic_fd,
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
    main(parser.parse_args())