import argparse
import numpy as np
import lmdb
import pickle
import random
from pathlib import Path
from tqdm import tqdm
import evaluation.utils


def extract_gt_from_lmdb(lmdb_path, max_clips=None):
    static_seqs, kine_seqs = [], []
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        clips_loaded = 0
        for key, value in tqdm(cursor, desc=f"Loading GT from {Path(lmdb_path).name}"):
            entry = pickle.loads(value)
            
            gt_3d = evaluation.utils.convert_6d_to_3d(entry["gt"])
                
            dim = gt_3d.shape[-1]
            static_seqs.append(gt_3d.reshape(-1, dim))
            kine_seqs.append((gt_3d[1:] - gt_3d[:-1]).reshape(-1, dim))
            
            clips_loaded += 1
            if max_clips and clips_loaded >= max_clips:
                break
    env.close()
    return static_seqs, kine_seqs

def main(args):
    print("Extracting Test Set Ground Truth...")
    test_static_seqs, test_kine_seqs = extract_gt_from_lmdb(args.test_lmdb)
    
    print(f"Extracting Train Set Ground Truth (Capped at {args.num_train_clips} clips)...")
    train_static_seqs, train_kine_seqs = extract_gt_from_lmdb(args.train_lmdb, max_clips=args.num_train_clips)
    
    # Pooled Diversity
    test_pool_s = np.vstack(test_static_seqs)
    test_pool_k = np.vstack(test_kine_seqs)
    
    div_samples_g = min(10000, test_pool_s.shape[0])
    div_samples_k = min(10000, test_pool_k.shape[0])
    
    static_div = float(evaluation.utils.calculate_diversity_np(test_pool_s, div_samples_g))
    kinematic_div = float(evaluation.utils.calculate_diversity_np(test_pool_k, div_samples_k))
    
    # Sequence-Averaged FD
    test_stats_s = [evaluation.utils.calculate_activation_statistics_np(seq) for seq in test_static_seqs]
    test_stats_k = [evaluation.utils.calculate_activation_statistics_np(seq) for seq in test_kine_seqs]
    train_stats_s = [evaluation.utils.calculate_activation_statistics_np(seq) for seq in train_static_seqs]
    train_stats_k = [evaluation.utils.calculate_activation_statistics_np(seq) for seq in train_kine_seqs]
    
    seq_fd_s, seq_fd_k = [], []
    for i in range(len(test_stats_s)):
        train_idx = random.randint(0, len(train_stats_s) - 1)
        
        m1_s, c1_s = test_stats_s[i]
        m2_s, c2_s = train_stats_s[train_idx]
        seq_fd_s.append(evaluation.utils.calculate_frechet_distance_np(m1_s, c1_s, m2_s, c2_s))
        
        m1_k, c1_k = test_stats_k[i]
        m2_k, c2_k = train_stats_k[train_idx]
        seq_fd_k.append(evaluation.utils.calculate_frechet_distance_np(m1_k, c1_k, m2_k, c2_k))
        
    print("\n" + "="*40)
    print("FINAL GT RANDOM BASELINE (TEST vs TRAIN)")
    print("="*40)
    print(f"Static Diversity: {static_div:.4f}")
    print(f"Kinematic Diversity: {kinematic_div:.4f}")
    print(f"Static FD:  {np.mean(seq_fd_s):.4f}")
    print(f"Kinematic FD: {np.mean(seq_fd_k):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_lmdb", type=str, required=True, help="Path to test set LMDB")
    parser.add_argument("--train_lmdb", type=str, required=True, help="Path to the training set LMDB")
    parser.add_argument("--num_train_clips", type=int, default=100, help="Number of random training clips to sample for the baseline")
    main(parser.parse_args())