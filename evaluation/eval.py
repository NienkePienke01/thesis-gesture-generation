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

        chunk_samples = 5 * 60 * sr 
        total_chunks = int(np.ceil(len(y) / chunk_samples))
        
        S_list, rms_list, oenv_list = [], [], []
        onset_raw_list, onset_bt_list, onset_bt_rms_list = [], [], []
        frame_offset = 0

        print(f"[2/3] Processing {total_chunks} chunks through original librosa pipeline...")
        for i in range(total_chunks):
            start_time = time.time()
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(y))
            y_chunk = y[start_idx:end_idx]
            
            # Exact original calculations
            oenv_chunk = librosa.onset.onset_strength(y=y_chunk, sr=sr)
            S_chunk = np.abs(librosa.stft(y=y_chunk))
            rms_chunk = librosa.feature.rms(S=S_chunk)
            
            onset_raw_chunk = librosa.onset.onset_detect(onset_envelope=oenv_chunk, backtrack=False)
            
            # Backtrack and apply the global frame offset
            if len(onset_raw_chunk) > 0:
                onset_bt_chunk = librosa.onset.onset_backtrack(onset_raw_chunk, oenv_chunk)
                onset_bt_rms_chunk = librosa.onset.onset_backtrack(onset_raw_chunk, rms_chunk[0])
                
                onset_raw_list.append(onset_raw_chunk + frame_offset)
                onset_bt_list.append(onset_bt_chunk + frame_offset)
                onset_bt_rms_list.append(onset_bt_rms_chunk + frame_offset)
            
            S_list.append(S_chunk)
            rms_list.append(rms_chunk)
            oenv_list.append(oenv_chunk)
            
            frame_offset += len(oenv_chunk)
            del y_chunk
            gc.collect()
            print(f"      -> Chunk {i+1:02d}/{total_chunks} completed in {time.time() - start_time:.2f}s")

        print("[3/3] Stitching arrays to finalize alignment object...")
        self.S = np.concatenate(S_list, axis=1)
        self.rms = np.concatenate(rms_list, axis=1)
        self.oenv = np.concatenate(oenv_list, axis=0)
        self.times = librosa.times_like(self.oenv, sr=sr)
        
        if not onset_raw_list:
            return None, None, None
            
        onset_raw = np.concatenate(onset_raw_list)
        onset_bt = np.concatenate(onset_bt_list)
        onset_bt_rms = np.concatenate(onset_bt_rms_list)
        
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
    audio_data = results["audio"]
    
    print("="*40)
    print("1. CALCULATING BEAT ALIGNMENT (6D Space)")
    print("="*40)
    
    align = Alignment(sigma=3, order=4)
    onset_raw, onset_bt, onset_bt_rms = align.load_audio(wave=audio_data, without_file=True, sr_audio=16000)
    
    if onset_raw is None:
        print("BA Score: No audio beats detected!")
    else:
        beats = align.load_pose(pose=pred_data, pose_fps=30, without_file=True)
        ba_score = align.calculate_align(onset_raw, onset_bt, onset_bt_rms, *beats, pose_fps=30, sr_audio=16000)
        print(f"Beat Alignment Score (BA): {ba_score:.4f}")

    print("\n" + "="*40)
    print("2. CALCULATING FGD & DIVERSITY (3D Joint Space)")
    print("="*40)
    
    pred_3d = convert_6d_to_3d(pred_data)
    gt_3d = convert_6d_to_3d(gt_data)
    
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
    # (Matches Audio2Photoreal reference: var across time/frames)
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