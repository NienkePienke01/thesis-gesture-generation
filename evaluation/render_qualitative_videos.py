import os
import shutil
import argparse
import math
import torch
from pathlib import Path
from tqdm import tqdm

# Import from your existing project structure
from data_processing.seamless_dataset import SeamlessDataset
import data_processing.utils as utils
from utils import rotation_conversions as rc

# Import generation and visualization tools
from visualisation.visualise_stage_2 import (
    load_stage2_and_stage1_pipeline, 
    reconstruct_and_render, 
    setup_reconstruction, 
    generate_motion_sequence, 
    _stage2_runtime_cache
)
import visualisation.reconstruction_tab_whole_body as stage_1_reconstruction


def dummy_progress(val, desc=""):
    """Overrides Gradio's gr.Progress() to keep console output clean."""
    pass


def generate_person_loader(npz_path, args):
    """
    Runs the full Stage 2 -> Stage 1 pipeline for a single person's NPZ file
    and returns a SeamlessSMPLHLoader containing their *generated* poses.
    """
    device = torch.device(args.device)
    
    # Setup Data
    setup_data = setup_reconstruction(npz_path, device)
    if isinstance(setup_data, tuple) and len(setup_data) == 2:
        raise ValueError(f"Setup failed: {setup_data[1]}")
        
    original_loader, segment_data, partner_segment_data, relationship = setup_data
    
    stride = _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_frames = _stage2_runtime_cache["segmenter"].pose_length
    num_frames = segment_data.aligned_pose.shape[0]
    num_chunks = math.floor((num_frames - chunk_frames) / stride) + 1
    batch_size = 1

    # Generate Stage 2 Tokens
    pose_codes = generate_motion_sequence(
        device, segment_data, partner_segment_data, 
        chunk_frames, stride, num_chunks, batch_size, 
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, warm_start=args.warm_start
    )

    # Decode Stage 1 Tokens to 6D Pose
    downsample_factor = _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_codes = (chunk_frames // downsample_factor)
    dec_stride = _stage2_runtime_cache["segmenter"].stride 

    token_chunks = pose_codes.unfold(dimension=1, size=chunk_codes, step=dec_stride // downsample_factor)
    token_chunks = token_chunks.permute(1, 0, 3, 2)
    pose_codes_list = list(token_chunks.unbind(dim=0))

    num_dec_chunks = len(pose_codes_list)
    valid_frames = (num_dec_chunks - 1) * dec_stride + chunk_frames
    full_body_joint_count = 43
    
    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)
    pre_frames = chunk_frames - dec_stride
    weight_mask = torch.zeros((batch_size, valid_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=pre_frames, device=device).view(1, -1, 1, 1)

    final_6d_pose = stage_1_reconstruction.decode_sequence(
        device, pose_codes_list, chunk_frames, dec_stride, num_dec_chunks, 
        full_body_joint_count, batch_size, pre_frames, 
        final_reconstructed_6d, weight_mask, fade_in, fade_out, model=_stage2_runtime_cache["stage1_model"]
    )

    # Convert back to Axis-Angle and Overwrite a new Loader
    reconstructed_matrix = rc.rotation_6d_to_matrix(final_6d_pose)
    final_reconstructed_pose = rc.matrix_to_axis_angle(reconstructed_matrix).reshape(valid_frames, -1)

    generated_loader = original_loader.overwrite_with_reconstructed_pose(final_reconstructed_pose)
    
    return generated_loader


def main(args):
    # Gather file IDs
    file_ids = []
    if args.file_ids:
        file_ids.extend(args.file_ids)
    if args.file_ids_txt:
        with open(args.file_ids_txt, 'r') as f:
            file_ids.extend([line.strip() for line in f if line.strip()])
            
    if not file_ids:
        print("Error: No file IDs provided! Use --file_ids or --file_ids_txt.")
        return

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Loading dataset from {args.data_folder}...")
    dataset = SeamlessDataset(args.data_folder, file_ids=file_ids)
    print(f"Successfully mapped {len(dataset.samples)} out of {len(file_ids)} requested IDs.")

    print("\n>>> Initializing Stage 2 and Stage 1 Pipeline...")
    status = load_stage2_and_stage1_pipeline(
        args.stage2_config, 
        args.stage2_checkpoint, 
        args.segmenter_config, 
        args.device
    )
    print(status)

    # Inference & Rendering Loop
    print("\n" + "="*50)
    print(f"STARTING BATCH VIDEO RENDERING (Mode: {args.mode.upper()})")
    print("="*50)
    
    success_count = 0
    for file_id in tqdm(dataset.file_ids, desc="Processing Sequences"):
        npz_path = str(dataset.samples.get(file_id))
        
        if not npz_path or not os.path.exists(npz_path):
            print(f"\nWarning: {file_id} not found on disk. Skipping.")
            continue
            
        try:
            video_path = None
            msg = ""

            if args.mode == "comparison":
                # Person A: Ground Truth vs Generated side-by-side
                video_path, msg = reconstruct_and_render(
                    target_npz_file=npz_path, 
                    device=args.device, 
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    temperature=args.temperature, 
                    warm_start=args.warm_start, 
                    progress=dummy_progress
                )
                
            elif args.mode == "dyadic_gt":
                # Generated Person A vs Ground Truth Person B
                loader_args = {'device': args.device, 'load_vertices': True, 'with_meta_data': True}
                loader_1_gt, loader_2_gt = dataset.get_interaction_pair(file_id, loader_args)
                
                if loader_2_gt is None:
                    msg = "Partner data not found. Cannot render a dyadic interaction."
                else:
                    # Generate Person A
                    print(f"\nGenerating Person A ({loader_1_gt.sample_name})...")
                    loader_1_gen = generate_person_loader(loader_1_gt.npz_path, args)
                    
                    # We DO NOT generate Person B. We keep loader_2_gt as the ground truth.
                    print(f"Using Ground Truth for Person B ({loader_2_gt.sample_name})...")
                    
                    # Render the generated avatar facing the real human
                    print("Rendering hybrid interaction video...")
                    video_path = utils.generate_interaction_video(
                        loader_1_gen, loader_2_gt, progress=dummy_progress, use_cache=False, include_audio=True
                    )
                    msg = "Generated vs GT Dyadic interaction rendered successfully."

            elif args.mode == "dyadic_generated":
                # Generated Dyad: AI Person A vs AI Person B
                loader_args = {'device': args.device, 'load_vertices': False, 'with_meta_data': True}
                loader_1_gt, loader_2_gt = dataset.get_interaction_pair(file_id, loader_args)

                if loader_2_gt is None:
                    msg = "Partner data not found. Cannot generate a dyadic interaction."
                else:
                    # Generate Person A
                    print(f"\nGenerating Person A ({loader_1_gt.sample_name})...")
                    loader_1_gen = generate_person_loader(loader_1_gt.npz_path, args)
                    torch.cuda.empty_cache() # Clear VRAM safely between passes
                    
                    # Generate Person B
                    print(f"Generating Person B ({loader_2_gt.sample_name})...")
                    loader_2_gen = generate_person_loader(loader_2_gt.npz_path, args)
                    torch.cuda.empty_cache()
                    
                    # Render them facing each other
                    print("Rendering full AI interaction video...")
                    video_path = utils.generate_interaction_video(
                        loader_1_gen, loader_2_gen, progress=dummy_progress, use_cache=False, include_audio=True
                    )
                    msg = "Generated Dyadic interaction rendered successfully."
            
            # File Management
            if video_path and isinstance(video_path, str) and os.path.exists(video_path):
                suffix_map = {
                    "comparison": "_comparison.mp4",
                    "dyadic_gt": "_hybrid_dyadic.mp4",
                    "dyadic_generated": "_ai_dyadic.mp4"
                }
                final_dest = output_dir / f"{file_id}{suffix_map[args.mode]}"
                shutil.copy(video_path, final_dest)
                success_count += 1
            else:
                print(f"\nFailed to generate video for {file_id}. Message: {msg}")
                
        except Exception as e:
            print(f"\nException occurred while processing {file_id}: {str(e)}")

    print("\n" + "="*50)
    print(f"FINISHED! Successfully rendered {success_count}/{len(dataset.samples)} videos.")
    print(f"Videos saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render qualitative comparison and dyadic videos.")
    
    # Run Mode
    parser.add_argument("--mode", type=str, choices=["comparison", "dyadic_gt", "dyadic_generated"], 
                        default="comparison", 
                        help="comparison: GT Person A vs Gen Person A. dyadic_gt: Gen Person A vs GT Person B. dyadic_generated: Gen Person A vs Gen Person B.")


    parser.add_argument("--stage2_config", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--segmenter_config", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True, help="Root folder of the Seamless dataset")
    parser.add_argument("--output_dir", type=str, default="qualitative_renders", help="Where to save the final MP4s")
    

    parser.add_argument("--file_ids", type=str, nargs='+', help="List of specific file IDs to render (space separated)")
    parser.add_argument("--file_ids_txt", type=str, help="Path to a .txt file containing one file ID per line")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--warm_start", action="store_true", help="Seed generation with ground truth context")
    
    main(parser.parse_args())