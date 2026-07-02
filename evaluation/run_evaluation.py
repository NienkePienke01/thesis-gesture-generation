import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Standalone Automated Evaluation Pipeline")

    parser.add_argument("--stage2_config", type=str, required=True, help="Path to the stage 2 config.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.bin)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")

    args = parser.parse_args()

    # Load the stage 2 config to get the evaluation parameters
    config_path = PROJECT_ROOT / args.stage2_config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Extract project name directly from the config
    project_name = config_dict.get("project", "unknown_project")

    # Ensure the evaluation block exists
    if "evaluation" not in config_dict:
        print(f"Error: No 'evaluation' block found in {args.stage2_config}")
        return

    # Convert the evaluation dictionary into a SimpleNamespace
    eval_args = SimpleNamespace(**config_dict["evaluation"])

    print("\n" + "="*50)
    print("STARTING STANDALONE EVALUATION PIPELINE")
    print(f"Project: {project_name}")
    print("="*50)

    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    explicit_video_render_path = PROJECT_ROOT / "qualitative_renders" / f"eval_auto_{project_name}_{eval_timestamp}"

    # Ensure output directories exist
    (PROJECT_ROOT / "evaluation_results").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "evaluation_metrics_results").mkdir(parents=True, exist_ok=True)
    explicit_video_render_path.mkdir(parents=True, exist_ok=True)

    # Handle single integer or list of chunk frames
    chunks = eval_args.chunk_frames if isinstance(eval_args.chunk_frames, list) else [eval_args.chunk_frames]

    for chunk in chunks:
        print(f"\n>>> Running Inference for chunk size: {chunk} frames")
        
        explicit_lmdb_path = PROJECT_ROOT / "evaluation_results" / f"eval_auto_{project_name}_chunk_{chunk}_{eval_timestamp}"
        explicit_metrics_path = PROJECT_ROOT / "evaluation_metrics_results" / f"metrics_auto_{project_name}_chunk_{chunk}_{eval_timestamp}.json"

        # Run Inference using prefixes and paths from the config
        infer_cmd_prefix = getattr(eval_args, "infer_cmd_prefix", ["python"])
        infer_cmd = infer_cmd_prefix + [
            "--stage2_config", args.stage2_config,
            "--stage2_checkpoint", args.checkpoint,
            "--segmenter_config", eval_args.segmenter_config,
            "--test_folder", eval_args.test_folder,
            "--chunk_frames", str(chunk),
            "--max_gb", str(getattr(eval_args, "max_gb", 10)),
            "--device", args.device,
            "--output_dir", str(explicit_lmdb_path),
            "-seed", str(eval_args.seed)
        ]
        
        print(f"Executing: {' '.join(infer_cmd)}")
        subprocess.run(infer_cmd, check=True)
        
        print(f"\n>>> Running Evaluation on {explicit_lmdb_path.name}")
        
        # Run Metric Evaluation
        eval_cmd_prefix = getattr(eval_args, "eval_cmd_prefix", ["python"])
        eval_cmd = eval_cmd_prefix + [
            "--results", str(explicit_lmdb_path),
            "--metrics_output", str(explicit_metrics_path),
            "--project_name", project_name,
            "-seed", str(eval_args.seed)
        ]
        
        print(f"Executing: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)

    # Generate Qualitative Videos
    print(f"\n>>> Generating videos in {explicit_video_render_path.name}")
    infer_videos_prefix = getattr(eval_args, "infer_videos_prefix", ["python"])
    infer_videos_cmd = infer_videos_prefix + [
        "--stage2_config", args.stage2_config,
        "--stage2_checkpoint", args.checkpoint,
        "--segmenter_config", eval_args.segmenter_config,
        "--data_folder", eval_args.test_folder,
        "--output_dir", str(explicit_video_render_path),
        "--warm_start",
        "--file_ids_txt", eval_args.file_ids,
        "--device", args.device,
        "-seed", str(eval_args.seed)
    ]
    
    print(f"Executing: {' '.join(infer_videos_cmd)}")
    subprocess.run(infer_videos_cmd, check=True)

    print("\n" + "="*50)
    print("EVALUATION PIPELINE COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()