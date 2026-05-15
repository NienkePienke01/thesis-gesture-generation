import lmdb
import pickle
import argparse
import random
from pathlib import Path
from loguru import logger
import json
from utils import rotation_conversions as rc
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cache_dir = PROJECT_ROOT / "cache" / "segmenter"
cache_dir_output = PROJECT_ROOT / "cache" / "training_data"

def split_source(args):
    sources = []
    for s in args.sources:
        name, train_percentage, validation_percentage = s.split(':')
        sources.append({
            "name": name, 
            "train_percentage": float(train_percentage), 
            "validation_percentage": float(validation_percentage),
            "path": cache_dir / name
        })

    output_train = cache_dir_output / f"{args.output_name}" / f"{args.output_name}_train"
    output_val = cache_dir_output / f"{args.output_name}" / f"{args.output_name}_val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)

    # Create the lmdbs
    if not args.stats_only:
        map_size = int(args.max_gb * 1024**3)
        train_env = lmdb.open(str(output_train), map_size=map_size)
        val_env = lmdb.open(str(output_val), map_size=map_size)

        train_idx = 0
        val_idx = 0
        manifest = {"sources": [], "total_train": 0, "total_val": 0}

        for src in sources:
            logger.info(f"Processing {src['name']}...")
            src_env = lmdb.open(str(src['path']), readonly=True, lock=False)
            
            with src_env.begin() as src_txn:
                # Get the unique fileid for each file and group the entries
                file_groups = {}
                cursor = src_txn.cursor()
                for key, value in cursor:
                    idx = int(key.decode('ascii'))
                    entry = pickle.loads(value)
                    f_id = entry["seamless_id"]
                    
                    if f_id not in file_groups:
                        file_groups[f_id] = []
                    file_groups[f_id].append(idx)
                
                seq_length = entry["pose"].shape[0]

                unique_ids = list(file_groups.keys())
                random.seed(args.seed)
                random.shuffle(unique_ids)

                total_entries = src_txn.stat()['entries']
                train_limit = int(total_entries * src['train_percentage'])
                val_limit = int(total_entries * src['validation_percentage'])
                
                train_indices = []
                validation_indices = []
                train_file_ids = []
                val_file_ids = []

                # Setup the train and val indices
                current_count = 0
                for f_id in unique_ids:
                    indices = file_groups[f_id]
                    # Sort indices to ensure frames appear in order
                    indices.sort() 
                    
                    if current_count < train_limit:
                        train_indices.extend(indices)
                        train_file_ids.append(f_id)
                    elif current_count < (train_limit + val_limit):
                        validation_indices.extend(indices)
                        val_file_ids.append(f_id)
                    
                    current_count += len(indices)

                # Write Train portion
                with train_env.begin(write=True) as t_txn:
                    for idx in train_indices:
                        data = src_txn.get(f"{idx:08d}".encode("ascii"))
                        t_txn.put(f"{train_idx:08d}".encode("ascii"), data)
                        train_idx += 1

                # Write validation portion
                with val_env.begin(write=True) as v_txn:
                    for idx in validation_indices:
                        data = src_txn.get(f"{idx:08d}".encode("ascii"))
                        v_txn.put(f"{val_idx:08d}".encode("ascii"), data)
                        val_idx += 1

                manifest["sources"].append({
                    "name": src["name"],
                    "requested_train_percent": src["train_percentage"],
                    "requested_val_percent": src["validation_percentage"],
                    "actual_train_count": len(train_indices),
                    "actual_val_count": len(validation_indices),
                    "train_file_ids": train_file_ids,
                    "val_file_ids": val_file_ids
                })
            src_env.close()

        manifest["total_train"] = train_idx
        manifest["total_val"] = val_idx
        train_hours = (train_idx * seq_length) / (30 * 3600)
        val_hours = (val_idx * seq_length) / (30 * 3600)
        manifest["total_train_hours"] = train_hours
        manifest["total_val_hours"] = val_hours

        with open(cache_dir_output / f"{args.output_name}" / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
        
        train_env.close()
        val_env.close()
        logger.info(f"Done! Combined total: {train_idx} Train | {val_idx} Val")
    logger.info(f"Computing the STD and Mean of the data")
    
    #----------
    # Calculate mean and std
    #----------
    all_data_in_6d = []
    temp_env = lmdb.open(str(output_train), readonly=True, lock=False)    
    with temp_env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            entry = pickle.loads(value)
            pose_axis_angles = torch.from_numpy(entry["pose"]).float()
            if i == 0:
                frames, dimension = pose_axis_angles.shape
                joint_count = dimension // 3
            pose_matrix = rc.axis_angle_to_matrix(pose_axis_angles.reshape(frames, joint_count, 3))
            pose_6d = rc.matrix_to_rotation_6d(pose_matrix).reshape(frames, joint_count * 6)
            all_data_in_6d.append(pose_6d)
        
        all_data_in_6d = torch.cat(all_data_in_6d, dim=0)

        mean = torch.mean(all_data_in_6d, dim=0)
        std = torch.std(all_data_in_6d, dim=0)
        stats = {
            "mean": mean,
            "std": std
        }
        save_path = cache_dir_output / f"{args.output_name}" / "mean_and_std.pt"
        torch.save(stats, save_path)
        logger.info(f"Mean {mean} and STD {std} saved to {save_path}")
    
    #-----
    # Verify dataset
    #-----
    logger.info("Verifying continuity in the new training set")
    check_env = lmdb.open(str(output_train), readonly=True, lock=False)
    with check_env.begin() as txn:
        cursor = txn.cursor()
        last_file = None
        last_seg_idx = -1
        continuity_breaks = 0

        total_entries = txn.stat()['entries']
        check_entries = min([500, total_entries])
        
        # Check first check_entries entries for continuity
        for i, (key, value) in enumerate(cursor):
            if i > check_entries: break 
            entry = pickle.loads(value)
            current_file = entry["seamless_id"]
            current_seg_idx = entry["segment_index"]
            
            if current_file == last_file:
                # If it's the same file, the segment index should be sequential
                if current_seg_idx != last_seg_idx + 1:
                    continuity_breaks += 1
            
            last_file = current_file
            last_seg_idx = current_seg_idx
            
    if continuity_breaks == 0:
        logger.success("Verification Passed: Segments are stored in contiguous sequences.")
    else:
        logger.warning(f"Verification Warning: Found {continuity_breaks} breaks in temporal sequence.")
    check_env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Example command args: --sources folder1:0.2:0.05 folder2:0.1:0.02
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--max_gb", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_only",action="store_true", dest="stats_only", default=False, help="To only compute the stats (AFTER you already computed the dataset before)")
    split_source(parser.parse_args())