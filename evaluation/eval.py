import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import librosa
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
import gc
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- SKELETON CONFIGURATION ---
L_SHOULDER, L_ELBOW, L_WRIST = 16, 18, 20
R_SHOULDER, R_ELBOW, R_WRIST = 17, 19, 21
# ------------------------------

class Alignment(object):
    """Class to calculate alignment between audio and pose data"""
    def __init__(self, sigma, order):
        self.sigma = sigma
        self.order = order
        self.times = self.oenv = self.S = self.rms = None

    def load_audio(self, wave, t_start=None, t_end=None, without_file=False, sr_audio=16000):
        print("[1/3] Formatting audio array...")
        if without_file:
            y, sr = wave, sr_audio
        else:
            y, sr = librosa.load(wave, sr=sr_audio)

        short_y = y  # [int(t_start*sr):int(t_end*sr)]
        self.oenv = librosa.onset.onset_strength(y=short_y, sr=sr)
        self.times = librosa.times_like(self.oenv)
        # Detect events without backtracking
        onset_raw = librosa.onset.onset_detect(
            onset_envelope=self.oenv, backtrack=False
        )
        if len(onset_raw) == 0:
            # print(len(wave))
            return None, None, None
        onset_bt = librosa.onset.onset_backtrack(onset_raw, self.oenv)
        self.S = np.abs(librosa.stft(y=short_y))
        self.rms = librosa.feature.rms(S=self.S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, self.rms[0])
        return onset_raw, onset_bt, onset_bt_rms

    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        return vel[0] / pose_fps + offset

    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2) / (2 * sigma**2))
        return dis_all_b2a / len(b) if len(b) > 0 else 0

    def calculate_align(self, onset_raw, onset_bt, onset_bt_rms, beat_right_arm, 
                        beat_right_shoulder, beat_right_wrist, beat_left_arm, 
                        beat_left_shoulder, beat_left_wrist, pose_fps=30, sr_audio=16000):
        audio_bt = librosa.frames_to_time(onset_bt_rms, sr=sr_audio)
        pose_bt = self.motion_frames2time(beat_right_wrist, 0, pose_fps)
        return self.GAHR(pose_bt, audio_bt, self.sigma)


def calculate_diversity(activation: np.ndarray, diversity_times: int = 10_000) -> float:
    assert len(activation.shape) == 2
    if activation.shape[0] <= diversity_times:
        diversity_times = activation.shape[0] // 2 
    
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist

def calculate_activation_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def convert_6d_to_3d(pose_6d: np.ndarray, device='cuda') -> np.ndarray:
    frames = pose_6d.shape[0]
    
    # Convert to Torch Tensor and Reshape
    pose_6d_tensor = torch.tensor(pose_6d, dtype=torch.float32, device=device).reshape(frames, 43, 6)
    
    #Convert 6D -> Matrix -> Axis-Angle
    pose_axis_angle = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(pose_6d_tensor))
    
    # Map the 43 joints back to full SMPL-H 52-joint structure
    body_mask = [0, 1, 3, 4, 6, 7, 9, 10]
    keep_indices = [i for i in range(21) if i not in body_mask]
    
    full_body_pose = torch.zeros((frames, 21, 3), device=device)
    full_body_pose[:, keep_indices, :] = pose_axis_angle[:, :13, :]
    left_hand_pose = pose_axis_angle[:, 13:28, :]
    right_hand_pose = pose_axis_angle[:, 28:43, :]
    global_orient = torch.zeros((frames, 1, 3), device=device)
    
    smplh = get_smplh_model(device, gender='male')
    all_joints_3d = []
    
    # Extract the 43 specific 3D joint positions (ignoring root and lower body)
    eval_indices = [idx + 1 for idx in keep_indices] + list(range(22, 52))
    
    with torch.no_grad():
        for i in range(0, frames, 512):
            end = min(i + 512, frames)
            curr_batch_size = end - i
            
            output = smplh(
                global_orient=global_orient[i:end].reshape(curr_batch_size, -1),
                body_pose=full_body_pose[i:end].reshape(curr_batch_size, -1),
                left_hand_pose=left_hand_pose[i:end].reshape(curr_batch_size, -1),
                right_hand_pose=right_hand_pose[i:end].reshape(curr_batch_size, -1),
                betas=smplh.betas.expand(curr_batch_size, -1),  # <-- The required fix
                return_verts=False
            )
            
            joints_3d = output.joints[:, eval_indices, :]
            all_joints_3d.append(joints_3d.cpu().numpy())
            
    return np.concatenate(all_joints_3d, axis=0).reshape(frames, -1)


def main(args):
    results = np.load(args.results, allow_pickle=True).item()
    
    pred_data = results["motion"] # 6D Rotations
    gt_data = results["gt"]       # 6D Rotations
    audio_data = np.array(results["audio"]).flatten().astype(np.float32)
    
    pose_fps = 30
    audio_sr = 16000
    chunk_seconds = 60 
    
    pose_chunk_size = chunk_seconds * pose_fps
    
    total_pose_frames = pred_data.shape[0]
    
    # Storage for aggregated metrics
    ba_scores = []
    all_pred_3d = []
    all_gt_3d = []

    print("="*40)
    print("1. CALCULATING BEAT ALIGNMENT & 3D CONVERSION (Chunked)")
    print("="*40)

    # Process audio in chunks
    for pose_start in range(0, total_pose_frames, pose_chunk_size):
        pose_end = min(pose_start + pose_chunk_size, total_pose_frames)
        audio_start = int((pose_start / pose_fps) * audio_sr)
        audio_end = int((pose_end / pose_fps) * audio_sr)

        pred_chunk = pred_data[pose_start:pose_end]
        gt_chunk = gt_data[pose_start:pose_end]
        audio_chunk = audio_data[audio_start:audio_end]
        
        align = Alignment(sigma=3, order=4)
        onset_raw, onset_bt, onset_bt_rms = align.load_audio(wave=audio_chunk, without_file=True, sr_audio=audio_sr)
        
        if onset_raw is not None:
            beats = align.load_pose(pose=pred_chunk, pose_fps=pose_fps, without_file=True)
            chunk_ba = align.calculate_align(onset_raw, onset_bt, onset_bt_rms, *beats, pose_fps=pose_fps, sr_audio=audio_sr)
            ba_scores.append(chunk_ba)
            
        # Convert 6D to 3D and store for global metrics later
        all_pred_3d.append(convert_6d_to_3d(pred_chunk))
        all_gt_3d.append(convert_6d_to_3d(gt_chunk))
        
        print(f"Processed chunk {pose_start//pose_chunk_size + 1}...")

    # Finalize BA Score
    final_ba_score = np.mean(ba_scores) if ba_scores else 0.0
    print(f"Final Average Beat Alignment Score (BA): {final_ba_score:.4f}")

    print("\n" + "="*40)
    print("2. CALCULATING GLOBAL FGD & DIVERSITY (3D Joint Space)")
    print("="*40)
    
    # Concatenate all 3D chunks into single massive arrays for statistical math
    pred_3d = np.concatenate(all_pred_3d, axis=0)
    gt_3d = np.concatenate(all_gt_3d, axis=0)
    
    feature_dim = pred_3d.shape[-1]

    # Reshape for static calculations [Frames, Features]
    pred_pose_last = pred_3d.reshape(-1, feature_dim)
    gt_pose_last = gt_3d.reshape(-1, feature_dim)

    # Static Diversity
    var_g = calculate_diversity(pred_pose_last)
    print(f"Static Diversity (var_g): {var_g.mean():.4f}")
    
    # Static FGD
    pred_mu_g, pred_cov_g = calculate_activation_statistics(pred_pose_last)
    gt_mu_g, gt_cov_g = calculate_activation_statistics(gt_pose_last)
    fid_g = calculate_frechet_distance(gt_mu_g, gt_cov_g, pred_mu_g, pred_cov_g)
    print(f"Static FGD (fid_g): {fid_g:.4f}")
    
    # Kinematic processing (Velocity)
    pred_motion = pred_3d[1:, :] - pred_3d[:-1, :]
    gt_motion = gt_3d[1:, :] - gt_3d[:-1, :]
    
    pred_motion_last = pred_motion.reshape(-1, feature_dim)
    gt_motion_last = gt_motion.reshape(-1, feature_dim)
    
    # Kinematic Diversity 
    var_k = np.var(pred_3d, axis=0) 
    print(f"Kinematic Diversity (var_k): {var_k.mean():.4f}")

    # Kinematic FGD
    pred_mu_k, pred_cov_k = calculate_activation_statistics(pred_motion_last)
    gt_mu_k, gt_cov_k = calculate_activation_statistics(gt_motion_last)
    fid_k = calculate_frechet_distance(gt_mu_k, gt_cov_k, pred_mu_k, pred_cov_k)
    print(f"Kinematic FGD (fid_k): {fid_k:.4f}")

    # Save the metrics
    input_path = Path(args.results)
    project_name = input_path.stem.split('-')[1] if len(input_path.stem.split('-')) > 1 else "unknown"

    metrics = {
        "project_name": project_name,
        "ba_score": float(ba_score) if 'ba_score' in locals() else None,
        "static_diversity": float(var_g.mean()),
        "kinematic_diversity": float(var_k.mean()), 
        "static_fgd": float(fid_g),
        "kinematic_fgd": float(fid_k)
    }

    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path("evaluation_metrics_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"metrics_{project_name}_{start_time}.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved evaluation metrics for '{project_name}' to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    main(args)