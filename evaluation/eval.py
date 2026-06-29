import os
import librosa
from evaluation.utils import peak_pick
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
        self.pose_data = []

    def load_audio(self, wave, t_start=0, t_end=None, without_file=False, sr_audio=16000):
        if without_file:
            y = wave
            sr = sr_audio
        else:
            y, sr = librosa.load(wave, sr=sr_audio)
            
        end_idx = int(t_end * sr) if t_end else len(y)
        short_y = y[int(t_start * sr):end_idx] 
        
        self.oenv = librosa.onset.onset_strength(y=short_y, sr=sr)
        self.times = librosa.times_like(self.oenv)
        onset_raw = librosa.onset.onset_detect(onset_envelope=self.oenv, backtrack=False)
        if len(onset_raw) == 0:
            return None, None, None
        onset_bt = librosa.onset.onset_backtrack(onset_raw, self.oenv)
        self.S = np.abs(librosa.stft(y=short_y))
        self.rms = librosa.feature.rms(S=self.S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, self.rms[0])
        return onset_raw, onset_bt, onset_bt_rms

    def load_pose(self, pose, t_start, t_end, pose_fps):
        start_idx = int(t_start * pose_fps)
        end_idx = int(t_end * pose_fps) if t_end else len(pose)
        data_each_file = pose[start_idx:end_idx]  
        vel = data_each_file[1:, :] - data_each_file[:-1, :]
        
        vel_right_shoulder = np.linalg.norm(np.array([vel[:, 9 * 3], vel[:, 9 * 3 + 1], vel[:, 9 * 3 + 2]]), axis=0)
        vel_right_arm = np.linalg.norm(np.array([vel[:, 10 * 3], vel[:, 10 * 3 + 1], vel[:, 10 * 3 + 2]]), axis=0)
        vel_right_wrist = np.linalg.norm(np.array([vel[:, 11 * 3], vel[:, 11 * 3 + 1], vel[:, 11 * 3 + 2]]), axis=0)
        beat_right_arm = argrelextrema(vel_right_arm, np.less, order=self.order)
        beat_right_shoulder = argrelextrema(vel_right_shoulder, np.less, order=self.order)
        beat_right_wrist = argrelextrema(vel_right_wrist, np.less, order=self.order)
        
        vel_left_shoulder = np.linalg.norm(np.array([vel[:, 5 * 3], vel[:, 5 * 3 + 1], vel[:, 5 * 3 + 2]]), axis=0)
        vel_left_arm = np.linalg.norm(np.array([vel[:, 6 * 3], vel[:, 6 * 3 + 1], vel[:, 6 * 3 + 2]]), axis=0)
        vel_left_wrist = np.linalg.norm(np.array([vel[:, 7 * 3], vel[:, 7 * 3 + 1], vel[:, 7 * 3 + 2]]), axis=0)
        beat_left_arm = argrelextrema(vel_left_arm, np.less, order=self.order)
        beat_left_shoulder = argrelextrema(vel_left_shoulder, np.less, order=self.order)
        beat_left_wrist = argrelextrema(vel_left_wrist, np.less, order=self.order)
        
        return (beat_right_arm, beat_right_shoulder, beat_right_wrist,
                beat_left_arm, beat_left_shoulder, beat_left_wrist)

    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        pose_beats = self.load_pose(pose, t_start, t_end, pose_fps)
        return onset_raw, onset_bt, onset_bt_rms, *pose_beats

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beats = self.load_pose(pose, i, i + dur, pose_fps)
            beat_right_wrist = beats[2]
            dis_all_b2a = self.calculate_align(onset_bt_rms, beat_right_wrist, pose_fps)
            print(f"{i}s: ", dis_all_b2a)

    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max), y_axis="log", x_axis="time", ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label="Onset strength")
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label="Raw onsets", color="r")
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label="Onset strength")
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label="Backtracked", color="r")
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label="RMS")
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label="Backtracked (RMS)", color="r")
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)

    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel[0] / pose_fps + offset
        return time_vel

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
        dis_all_b2a /= len(b)
        return dis_all_b2a

    def calculate_align(self, onset_bt_rms, beat_right_wrist, pose_fps=25):
        audio_bt = librosa.frames_to_time(onset_bt_rms)
        pose_bt = self.motion_frames2time(beat_right_wrist, 0, pose_fps)
        avg_dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
        return avg_dis_all_b2a



def calculate_diversity(activation: np.ndarray, diversity_times: int = 10_000) -> float:
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
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

import random

import random

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
            
            all_gt_static_stats.append(calculate_activation_statistics(gt_static))
            all_gt_kine_stats.append(calculate_activation_statistics(gt_kine))
            
            # Ensure diversity_times is strictly smaller than the number of frames
            div_samples_g = min(10000, gt_static.shape[0] - 1)
            div_samples_k = min(10000, gt_kine.shape[0] - 1)
            
            all_gt_static_div.append(calculate_diversity(gt_static, diversity_times=div_samples_g).mean())
            all_gt_kine_div.append(calculate_diversity(gt_kine, diversity_times=div_samples_k).mean())

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
        gt_random_static_fd.append(calculate_frechet_distance(mu1_g, cov1_g, mu2_g, cov2_g) * 10.0) # Scaled for readability
        
        # Kinematic FD
        mu1_k, cov1_k = all_gt_kine_stats[i]
        mu2_k, cov2_k = all_gt_kine_stats[j]
        gt_random_kine_fd.append(calculate_frechet_distance(mu1_k, cov1_k, mu2_k, cov2_k))
        
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
    static_gt_div, static_gt_random_fd, kine_gt_div, kine_gt_random_fd = calculate_gt_baselines(env) # GT values
    pose_fps = 30
    audio_sr = 16000
    
    print("="*40)
    print("1. EVALUATING CLIPS FROM LMDB (Per-Sequence)")
    print("="*40)
    
    all_static_div, all_kinematic_div = [], []
    all_static_fd, all_kinematic_fd = [], []
    # align = Alignment(sigma=3, order=4)
    
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Evaluating Clips"):
            entry = pickle.loads(value)

            # onset_raw, onset_bt, onset_bt_rms = align.load_audio(wave=audio_data, without_file=True, sr_audio=audio_sr)
                
            # if onset_raw is not None:
            #     beats = align.load_pose(pose=pred_data, t_start=0, t_end=None, pose_fps=pose_fps)
            #     beat_right_wrist = beats[2] 
            #     ba_score = align.calculate_align(onset_bt_rms, beat_right_wrist, pose_fps=pose_fps)
            
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
            all_static_div.append(calculate_diversity(pred_static, diversity_times=div_samples_g).mean())
            all_kinematic_div.append(calculate_diversity(pred_kine, diversity_times=div_samples_k).mean())
            
            # Per-sequence FD
            pred_mu_g, pred_cov_g = calculate_activation_statistics(pred_static)
            gt_mu_g, gt_cov_g = calculate_activation_statistics(gt_static)
            all_static_fd.append(calculate_frechet_distance(gt_mu_g, gt_cov_g, pred_mu_g, pred_cov_g) * 10.0) # Scaled for DyaDiT readability
            
            pred_mu_k, pred_cov_k = calculate_activation_statistics(pred_kine)
            gt_mu_k, gt_cov_k = calculate_activation_statistics(gt_kine)
            all_kinematic_fd.append(calculate_frechet_distance(gt_mu_k, gt_cov_k, pred_mu_k, pred_cov_k))
            
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