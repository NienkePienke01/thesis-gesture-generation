import librosa
import argparse
from scipy.signal import argrelextrema
import math
import numpy as np
from scipy import linalg
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

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
        if without_file:
            y = wave
            sr = sr_audio
        else:
            y, sr = librosa.load(wave, sr=sr_audio)
            
        short_y = y 
        self.oenv = librosa.onset.onset_strength(y=short_y, sr=sr)
        self.times = librosa.times_like(self.oenv, sr=sr) # Fixed SR mapping
        
        onset_raw = librosa.onset.onset_detect(onset_envelope=self.oenv, backtrack=False)
        if len(onset_raw) == 0:
            return None, None, None
            
        onset_bt = librosa.onset.onset_backtrack(onset_raw, self.oenv)
        self.S = np.abs(librosa.stft(y=short_y))
        self.rms = librosa.feature.rms(S=self.S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, self.rms[0])
        
        return onset_raw, onset_bt, onset_bt_rms

    def load_pose(self, pose, t_start=None, t_end=None, pose_fps=30, without_file=False):
        data_each_file = pose 
        vel = data_each_file[1:, :] - data_each_file[:-1, :]
        
        vel_right_shoulder = np.linalg.norm(vel[:, R_SHOULDER*6 : (R_SHOULDER+1)*6], axis=1)
        vel_right_arm = np.linalg.norm(vel[:, R_ELBOW*6 : (R_ELBOW+1)*6], axis=1)
        vel_right_wrist = np.linalg.norm(vel[:, R_WRIST*6 : (R_WRIST+1)*6], axis=1)
        
        beat_right_arm = argrelextrema(vel_right_arm, np.less, order=self.order)
        beat_right_shoulder = argrelextrema(vel_right_shoulder, np.less, order=self.order)
        beat_right_wrist = argrelextrema(vel_right_wrist, np.less, order=self.order)
        
        vel_left_shoulder = np.linalg.norm(vel[:, L_SHOULDER*6 : (L_SHOULDER+1)*6], axis=1)
        vel_left_arm = np.linalg.norm(vel[:, L_ELBOW*6 : (L_ELBOW+1)*6], axis=1)
        vel_left_wrist = np.linalg.norm(vel[:, L_WRIST*6 : (L_WRIST+1)*6], axis=1)
        
        beat_left_arm = argrelextrema(vel_left_arm, np.less, order=self.order)
        beat_left_shoulder = argrelextrema(vel_left_shoulder, np.less, order=self.order)
        beat_left_wrist = argrelextrema(vel_left_wrist, np.less, order=self.order)
        
        return (beat_right_arm, beat_right_shoulder, beat_right_wrist,
                beat_left_arm, beat_left_shoulder, beat_left_wrist)

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

    def calculate_align(self, onset_raw, onset_bt, onset_bt_rms, beat_right_arm, 
                        beat_right_shoulder, beat_right_wrist, beat_left_arm, 
                        beat_left_shoulder, beat_left_wrist, pose_fps=30, sr_audio=16000):
        # Explicitly pass sr_audio to prevent librosa from defaulting to 22050 and skewing timestamps
        audio_bt = librosa.frames_to_time(onset_bt_rms, sr=sr_audio)
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
            raise ValueError("Imaginary component {}".format(np.max(np.abs(covmean.imag))))
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def main(args):
    results = np.load(args.results, allow_pickle=True).item()
    
    # Load raw data
    pred_data = results["motion"]
    gt_data = results["gt"]
    audio_data = results["audio"]
    
    print("="*40)
    print("1. CALCULATING BEAT ALIGNMENT")
    print("="*40)
    # Instantiate Alignment
    align = Alignment(sigma=3, order=4)
    onset_raw, onset_bt, onset_bt_rms = align.load_audio(wave=audio_data, without_file=True, sr_audio=16000)
    
    if onset_raw is None:
        print("BA Score: No audio beats detected!")
    else:
        # We must pass the RAW 258-dimension pred_data here so indices 16-21 map properly
        beats = align.load_pose(pose=pred_data, pose_fps=30, without_file=True)
        ba_score = align.calculate_align(onset_raw, onset_bt, onset_bt_rms, *beats, pose_fps=30, sr_audio=16000)
        print(f"Beat Alignment Score (BA): {ba_score:.4f}")

    print("\n" + "="*40)
    print("2. CALCULATING FGD & DIVERSITY")
    print("="*40)
    
    # Slice the body to only evaluate upper body
    body_mask = [0, 1, 3, 4, 6, 7, 9, 10]
    total_joints = 43
    keep_joints = [i for i in range(total_joints) if i not in body_mask]
    
    keep_dims = []
    for j in keep_joints:
        keep_dims.extend(range(j * 6, (j + 1) * 6))
        
    feature_dim = len(keep_dims) 
    
    # Slice lower body out of FGD inputs
    pred_data = pred_data[..., keep_dims]
    gt_data = gt_data[..., keep_dims]

    pred_pose_last = pred_data.reshape(-1, feature_dim)
    gt_pose_last = gt_data.reshape(-1, feature_dim)

    # Diversity
    var_g = calculate_diversity(pred_pose_last)
    print(f"Static Diversity (var_g): {var_g.mean():.4f}")
    
    # Static FGD
    pred_mu_g, pred_cov_g = calculate_activation_statistics(pred_pose_last)
    gt_mu_g, gt_cov_g = calculate_activation_statistics(gt_pose_last)
    fid_g = calculate_frechet_distance(gt_mu_g, gt_cov_g, pred_mu_g, pred_cov_g)
    print(f"Static FGD (fid_g): {fid_g:.4f}")
    
    # Kinematic FGD
    pred_motion = pred_data[1:, :] - pred_data[:-1, :]
    gt_motion = gt_data[1:, :] - gt_data[:-1, :]
    
    pred_motion_last = pred_motion.reshape(-1, feature_dim)
    gt_motion_last = gt_motion.reshape(-1, feature_dim)
    
    pred_mu_k, pred_cov_k = calculate_activation_statistics(pred_motion_last)
    gt_mu_k, gt_cov_k = calculate_activation_statistics(gt_motion_last)
    fid_k = calculate_frechet_distance(gt_mu_k, gt_cov_k, pred_mu_k, pred_cov_k)
    print(f"Kinematic FGD (fid_k): {fid_k:.4f}")

    # Kinematic diversity
    var_k = calculate_diversity(pred_motion_last)
    print(f"Kinematic Diversity (var_k): {var_k.mean():.4f}")


    # Save the metrics
    input_path = Path(args.results)
    project_name = input_path.stem.split('-')[1] if len(input_path.stem.split('-')) > 1 else "unknown"

    # Collect results
    metrics = {
        "project_name": project_name,
        "ba_score": float(ba_score) if 'ba_score' in locals() else None,
        "static_diversity": float(var_g.mean()),
        "kinematic_diversity": float(var_k.mean()), # Added this
        "static_fgd": float(fid_g),
        "kinematic_fgd": float(fid_k)
    }

    # Save to JSON
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