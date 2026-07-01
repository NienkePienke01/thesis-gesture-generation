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
L_SHOULDER, L_ELBOW, L_WRIST = 15, 17, 19
R_SHOULDER, R_ELBOW, R_WRIST = 16, 18, 20
# ------------------------------

def main(args):
    env = lmdb.open(args.results, readonly=True)
    
    # the Beat Evaluator
    beat_evaluator = evaluation.utils.BeatAlignment(sr_audio=16000, pose_fps=30, sigma=0.1)
    
    print("\n" + "="*40)
    print("1. EVALUATING GENERATED CLIPS FROM LMDB")
    print("="*40)
    
    all_pred_s_seqs, all_pred_k_seqs = [], []
    all_gt_s_seqs, all_gt_k_seqs = [], []
    beat_scores = []
    beat_scores_gt = []
    
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Extracting Frames & Computing Metrics"):
            entry = pickle.loads(value)
            
            # Convert to 3D Joints
            pred_3d = evaluation.utils.convert_6d_to_3d(entry["motion"]) # Shape: (frames, joints, 3)
            gt_3d = evaluation.utils.convert_6d_to_3d(entry["gt"])

            # Calculate Beat Alignment for this sequence (Tracking Wrists)
            if "audio" in entry and entry["audio"] is not None and len(entry["audio"]) > 0:
                gahr_score = beat_evaluator.evaluate_sequence(
                    wave=entry["audio"], 
                    pose_3d=pred_3d, 
                    joint_indices=[L_WRIST, R_WRIST]
                )
                beat_scores.append(gahr_score)

                gahr_score_gt = beat_evaluator.evaluate_sequence(
                    wave=entry["audio"], 
                    pose_3d=gt_3d, 
                    joint_indices=[L_WRIST, R_WRIST]
                )
                beat_scores_gt.append(gahr_score_gt)

            # Format for Diversity and FD
            dim = pred_3d.shape[-1] * pred_3d.shape[-2] if len(pred_3d.shape) == 3 else pred_3d.shape[-1]
            
            all_pred_s_seqs.append(pred_3d.reshape(-1, dim))
            all_gt_s_seqs.append(gt_3d.reshape(-1, dim))
            
            all_pred_k_seqs.append((pred_3d[1:] - pred_3d[:-1]).reshape(-1, dim))
            all_gt_k_seqs.append((gt_3d[1:] - gt_3d[:-1]).reshape(-1, dim))
            
    env.close()

    # Pooled Diversity
    print("\nCalculating Diversity...")
    pooled_pred_s = np.vstack(all_pred_s_seqs)
    pooled_pred_k = np.vstack(all_pred_k_seqs)

    div_samples_s = min(10000, pooled_pred_s.shape[0])
    div_samples_k = min(10000, pooled_pred_k.shape[0])

    static_div = float(evaluation.utils.calculate_diversity_np(pooled_pred_s, div_samples_s))
    kinematic_div = float(evaluation.utils.calculate_diversity_np(pooled_pred_k, div_samples_k))

    # Sequence-Averaged FD
    print("Calculating Fréchet Distances...")
    seq_fd_s, seq_fd_k = [], []
    for i in tqdm(range(len(all_pred_s_seqs)), desc="FD Progress"):
        mp_s, cp_s = evaluation.utils.calculate_activation_statistics_np(all_pred_s_seqs[i])
        mg_s, cg_s = evaluation.utils.calculate_activation_statistics_np(all_gt_s_seqs[i])
        seq_fd_s.append(evaluation.utils.calculate_frechet_distance_np(mp_s, cp_s, mg_s, cg_s))
        
        mp_k, cp_k = evaluation.utils.calculate_activation_statistics_np(all_pred_k_seqs[i])
        mg_k, cg_k = evaluation.utils.calculate_activation_statistics_np(all_gt_k_seqs[i])
        seq_fd_k.append(evaluation.utils.calculate_frechet_distance_np(mp_k, cp_k, mg_k, cg_k))

    static_fd = float(np.mean(seq_fd_s))
    kinematic_fd = float(np.mean(seq_fd_k))
    avg_beat_score = float(np.mean(beat_scores)) if beat_scores else 0.0
    avg_gt_beat_score = float(np.mean(beat_scores_gt)) if beat_scores_gt else 0.0
    
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(f"Static Diversity:    {static_div:.4f}")
    print(f"Kinematic Diversity: {kinematic_div:.4f}")
    print(f"Static FD:           {static_fd:.4f}")
    print(f"Kinematic FD:        {kinematic_fd:.4f}")
    print(f"Beat Alignment GAHR: {avg_beat_score:.4f}")
    print(f"Beat Alignment GAHR GT: {avg_gt_beat_score:.4f}")
    
    input_path = Path(args.results)
    project_name = input_path.stem.split('-')[1] if len(input_path.stem.split('-')) > 1 else "unknown"

    metrics = {
        "project_name": project_name,
        "static_diversity": static_div,
        "kinematic_diversity": kinematic_div, 
        "static_FD": static_fd,
        "kinematic_FD": kinematic_fd,
        "beat_alignment_gahr": avg_beat_score,
        "beat_alignment_gahr_GT": avg_gt_beat_score
    }

    if args.metrics_output:
        output_file = Path(args.metrics_output)
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("evaluation_metrics_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = output_dir / f"metrics_{project_name}_{start_time}.json"
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved evaluation metrics for '{project_name}' to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, default=None)
    main(parser.parse_args())