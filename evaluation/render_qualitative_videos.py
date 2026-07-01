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

    # Re-initialize a loader and overwrite the pose
    generated_loader = utils.SeamlessSMPLHLoader(npz_path, device=device, with_meta_data=True)
    generated_loader.overwrite_with_reconstructed_pose(final_reconstructed_pose)
    
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

    # Prepare Output Directories
    base_output_dir = Path(args.output_dir)
    dir_comparison = base_output_dir / "comparison"
    dir_dyadic_gt = base_output_dir / "dyadic_gt"
    dir_dyadic_gen = base_output_dir / "dyadic_generated"
    
    for d in [dir_comparison, dir_dyadic_gt, dir_dyadic_gen]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Loading dataset from {args.data_folder}...")
    dataset = SeamlessDataset(args.data_folder, file_ids=file_ids)
    print(f"Successfully mapped {len(dataset.samples)} out of {len(file_ids)} requested IDs.")

    # Initialize Models
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
    print("STARTING BATCH VIDEO RENDERING (All 3 Types)")
    print("="*50)
    
    success_count = 0
    for file_id in tqdm(dataset.file_ids, desc="Processing Sequences"):
        try:
            print(f"\n--- Processing {file_id} ---")
            loader_args = {'device': args.device, 'load_vertices': True, 'with_meta_data': True}
            loader_1_gt, loader_2_gt = dataset.get_interaction_pair(file_id, loader_args)

            if loader_1_gt is None:
                print(f"Warning: Primary file {file_id} not found on disk. Skipping.")
                continue

            # GENERATE PERSON A
            print("1/5 Generating Person A...")
            loader_1_gen = generate_person_loader(loader_1_gt.npz_path, args)
            torch.cuda.empty_cache()

            # RENDER COMPARISON (Person A GT vs Gen)
            print("2/5 Rendering Comparison Video...")
            comp_path = utils.generate_reconstruction_comparison_video(
                loader_1_gt, loader_1_gen, progress=dummy_progress, use_cache=False, include_audio=True
            )
            shutil.copy(comp_path, dir_comparison / f"{file_id}_comparison.mp4")

            # Check if there is a partner to do dyadic interactions with
            if loader_2_gt is None:
                print("Partner data not found! Skipping Dyadic renders for this interaction.")
                success_count += 1
                continue

            # RENDER HYBRID DYADIC (Gen Person A vs GT Person B)
            print("3/5 Rendering Hybrid Dyadic Video (Gen A vs GT B)...")
            hybrid_path = utils.generate_interaction_video(
                loader_1_gen, loader_2_gt, progress=dummy_progress, use_cache=False
            )
            shutil.copy(hybrid_path, dir_dyadic_gt / f"{file_id}_hybrid_dyadic.mp4")

            # GENERATE PERSON B
            print("4/5 Generating Person B...")
            loader_2_gen = generate_person_loader(loader_2_gt.npz_path, args)
            torch.cuda.empty_cache()

            # RENDER FULL AI DYADIC (Gen Person A vs Gen Person B)
            print("5/5 Rendering Full AI Dyadic Video (Gen A vs Gen B)...")
            ai_path = utils.generate_interaction_video(
                loader_1_gen, loader_2_gen, progress=dummy_progress, use_cache=False
            )
            shutil.copy(ai_path, dir_dyadic_gen / f"{file_id}_ai_dyadic.mp4")
            
            success_count += 1
            print(f"Successfully finished all renders for {file_id}")
                
        except Exception as e:
            print(f"\nException occurred while processing {file_id}: {str(e)}")

    print("\n" + "="*50)
    print(f"FINISHED! Successfully processed {success_count}/{len(dataset.samples)} interactions.")
    print(f"Check the folders inside: {base_output_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render all qualitative analysis videos (Comparison, Hybrid Dyad, Full AI Dyad).")
    
    parser.add_argument("--stage2_config", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--segmenter_config", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True, help="Root folder of the Seamless dataset")
    parser.add_argument("--output_dir", type=str, default="qualitative_renders", help="Base directory to save the 3 video folders")
    
    parser.add_argument("--file_ids", type=str, nargs='+', help="List of specific file IDs to render (space separated)")
    parser.add_argument("--file_ids_txt", type=str, help="Path to a .txt file containing one file ID per line")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--warm_start", action="store_true")
    
    main(parser.parse_args())