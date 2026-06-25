"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import librosa
import librosa.display
import argparse
from scipy.signal import argrelextrema
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import math

import numpy as np
from scipy import linalg

class Alignment(object):
    """Class to calculate alignment between audio and pose data"""

    def __init__(self, sigma, order):
        self.sigma = sigma
        self.order = order
        self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []

    def load_audio(self, wave, t_start, t_end, without_file=False, sr_audio=16000):
        if without_file:
            y = wave
            sr = sr_audio
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

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        

        data_each_file = pose  # .reshape(-1, 189//3, 3)
        vel = data_each_file[1:, :] - data_each_file[:-1, :]
        
        # l2
        # breakpoint()
        vel_right_shoulder = np.linalg.norm(
            np.array([vel[:, 9 * 3], vel[:, 9 * 3 + 1], vel[:, 9 * 3 + 2]]), axis=0
        )
        vel_right_arm = np.linalg.norm(
            np.array([vel[:, 10 * 3], vel[:, 10 * 3 + 1], vel[:, 10 * 3 + 2]]), axis=0
        )
        vel_right_wrist = np.linalg.norm(
            np.array([vel[:, 11 * 3], vel[:, 11 * 3 + 1], vel[:, 11 * 3 + 2]]), axis=0
        )
        beat_right_arm = argrelextrema(vel_right_arm, np.less, order=self.order)
        beat_right_shoulder = argrelextrema(
            vel_right_shoulder, np.less, order=self.order
        )
        beat_right_wrist = argrelextrema(vel_right_wrist, np.less, order=self.order)
        vel_left_shoulder = np.linalg.norm(
            np.array([vel[:, 5 * 3], vel[:, 5 * 3 + 1], vel[:, 5 * 3 + 2]]), axis=0
        )
        vel_left_arm = np.linalg.norm(
            np.array([vel[:, 6 * 3], vel[:, 6 * 3 + 1], vel[:, 6 * 3 + 2]]), axis=0
        )
        vel_left_wrist = np.linalg.norm(
            np.array([vel[:, 7 * 3], vel[:, 7 * 3 + 1], vel[:, 7 * 3 + 2]]), axis=0
        )
        beat_left_arm = argrelextrema(vel_left_arm, np.less, order=self.order)
        beat_left_shoulder = argrelextrema(vel_left_shoulder, np.less, order=self.order)
        beat_left_wrist = argrelextrema(vel_left_wrist, np.less, order=self.order)
        return (
            beat_right_arm,
            beat_right_shoulder,
            beat_right_wrist,
            beat_left_arm,
            beat_left_shoulder,
            beat_left_wrist,
        )

    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        (
            beat_right_arm,
            beat_right_shoulder,
            beat_right_wrist,
            beat_left_arm,
            beat_left_shoulder,
            beat_left_wrist,
        ) = self.load_pose(pose, t_start, t_end, pose_fps)
        return (
            onset_raw,
            onset_bt,
            onset_bt_rms,
            beat_right_arm,
            beat_right_shoulder,
            beat_right_wrist,
            beat_left_arm,
            beat_left_shoulder,
            beat_left_wrist,
        )

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            (
                beat_right_arm,
                beat_right_shoulder,
                beat_right_wrist,
                beat_left_arm,
                beat_left_shoulder,
                beat_left_wrist,
            ) = self.load_pose(pose, i, i + dur, pose_fps)
            dis_all_b2a = self.calculate_align(
                onset_raw,
                onset_bt,
                onset_bt_rms,
                beat_right_arm,
                beat_right_shoulder,
                beat_right_wrist,
                beat_left_arm,
                beat_left_shoulder,
                beat_left_wrist,
            )
            print(f"{i}s: ", dis_all_b2a)

    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(
            librosa.amplitude_to_db(self.S, ref=np.max),
            y_axis="log",
            x_axis="time",
            ax=ax[0],
        )
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label="Onset strength")
        ax[1].vlines(
            librosa.frames_to_time(onset_raw),
            0,
            self.oenv.max(),
            label="Raw onsets",
            color="r",
        )
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label="Onset strength")
        ax[2].vlines(
            librosa.frames_to_time(onset_bt),
            0,
            self.oenv.max(),
            label="Backtracked",
            color="r",
        )
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label="RMS")
        ax[3].vlines(
            librosa.frames_to_time(onset_bt_rms),
            0,
            self.oenv.max(),
            label="Backtracked (RMS)",
            color="r",
        )
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)

    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel[0] / pose_fps + offset
        return time_vel

    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
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

    def calculate_align(
        self,
        onset_raw,
        onset_bt,
        onset_bt_rms,
        beat_right_arm,
        beat_right_shoulder,
        beat_right_wrist,
        beat_left_arm,
        beat_left_shoulder,
        beat_left_wrist,
        pose_fps=30,
    ):
        audio_bt = librosa.frames_to_time(onset_bt_rms)
        pose_bt = self.motion_frames2time(beat_right_wrist, 0, pose_fps)
        # avg_dis_all_b2a = self.GAHR(audio_bt, pose_bt, self.sigma)
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


def calculate_activation_statistics(
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def main(args):
    num_samples = 5
    results = np.load(args.results, allow_pickle=True).item()
    pred_reshaped = results["motion"].squeeze().reshape((num_samples, -1, 104, 600))
    gt_reshaped = results["gt"].squeeze().reshape((num_samples, -1, 104, 600))

    # calulate variance across the different samples generated
    cross_sample_var = np.var(pred_reshaped.reshape((num_samples, -1)), axis=0)
    print("cross var", cross_sample_var.mean())

    pred_pose_last = pred_reshaped.transpose((0, 1, 3, 2)).reshape(-1, 104)
    gt_pose_last = gt_reshaped.transpose((0, 1, 3, 2)).reshape(-1, 104)
    # calculate the static and kinematic diversity
    var_g = calculate_diversity(pred_pose_last)
    print("var_g", var_g.mean())
    var_k = np.var(pred_reshaped, axis=-1)
    print("var_k", var_k.mean())

    # calculate the static and kinematic fid
    pred_mu_g, pred_cov_g = calculate_activation_statistics(pred_pose_last)
    gt_mu_g, gt_cov_g = calculate_activation_statistics(gt_pose_last)
    fid_g = calculate_frechet_distance(gt_mu_g, gt_cov_g, pred_mu_g, pred_cov_g)
    print("fid_g", fid_g)
    # reshape for kinematic fid
    pred_motion = pred_reshaped[..., 1:] - pred_reshaped[..., :-1]
    gt_motion = gt_reshaped[..., 1:] - gt_reshaped[..., :-1]
    pred_motion_last = pred_motion.transpose((0, 1, 3, 2)).reshape(-1, 104)
    gt_motion_last = gt_motion.transpose((0, 1, 3, 2)).reshape(-1, 104)
    pred_mu_k, pred_cov_k = calculate_activation_statistics(pred_motion_last)
    gt_mu_k, gt_cov_k = calculate_activation_statistics(gt_motion_last)
    fid_k = calculate_frechet_distance(gt_mu_k, gt_cov_k, pred_mu_k, pred_cov_k)
    print("fid_k", fid_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    main(args)
