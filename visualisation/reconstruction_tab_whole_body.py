import gradio as gr
import torch
import numpy as np
import os
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import pyrender
import trimesh
import copy
from visualisation.file_browser import create_file_browser
from utils import rotation_conversions as rc
from models.rvq import RVQVAE
import data_processing.utils as utils
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import math

_stage1_model_cache = {
    "full_body_model": None,
    "normalize": False,
    "stats": None
}

def load_model_from_files(full_config, full_body_checkpoint, device):
    """
    Initializes model using the four explicitly uploaded files.
    """
    try:
        # Initialize Full Body Model
        with open(full_config, 'r') as f:
            full_config_dict = json.load(f)
        full_body_args = SimpleNamespace(**full_config_dict)
        full_body_args.data = SimpleNamespace(**full_config_dict["data"])
        full_body_model = RVQVAE(full_body_args, **full_config_dict["model_params"]).to(device)
        full_body_model.load_state_dict(torch.load(full_body_checkpoint, map_location=device))
        full_body_model.eval()

        _stage1_model_cache["full_body_model"] = full_body_model
        _stage1_model_cache["normalize"] = full_body_args.normalize
        if full_body_args.normalize:
            stats_path = Path(full_body_args.data.train_lmdb).parent / "mean_and_std.pt"
            full_body_args.stats_path = stats_path
            _stage1_model_cache["stats"] = torch.load(full_body_args.stats_path, map_location=device)
        
        return "Status: Full Body model loaded successfully."
    except Exception as e:
        return f"Status: Error loading model - {str(e)}"

def reconstruct_and_render(target_npz_file, device, progress=gr.Progress()):
    if _stage1_model_cache["full_body_model"] is None:
        return None, "Status: Please load the model first."

    original_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device)
    full_pose_sequence = original_loader.get_trainable_pose().to(device) # Shape: [Total_Frames, 129]
    
    chunk_frames = 64
    stride = 20
    num_frames = full_pose_sequence.shape[0]
    num_chunks = math.floor((num_frames - chunk_frames) / stride) + 1
    full_body_joint_count = 43
    batch_size = 1 # We do not batch here, so its just hard set to 1
    #
    valid_frames = (num_chunks - 1) * stride + chunk_frames
    latent_last = None
    pre_frames = chunk_frames - stride
    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)

    if _stage1_model_cache["normalize"]:
        norm_mean = _stage1_model_cache["stats"]["mean"]
        norm_std = _stage1_model_cache["stats"]["std"]
    
    with torch.no_grad():
        model = _stage1_model_cache["full_body_model"]
        for i in range(num_chunks):
            start_frame = i * stride
            end_frame = start_frame + chunk_frames
            chunk_tensor = full_pose_sequence[start_frame:end_frame].unsqueeze(0) # Shape: [1, chunk_frames, 129]
            
            # Full body
            full_body_target_pose = chunk_tensor

            if _stage1_model_cache["normalize"]:
                full_body_target_pose = (full_body_target_pose - norm_mean) / (norm_std) # normalise it

            full_body_target_pose_matrix = rc.axis_angle_to_matrix(full_body_target_pose.reshape(batch_size, chunk_frames, full_body_joint_count, 3))
            full_body_target_pose_6d = rc.matrix_to_rotation_6d(full_body_target_pose_matrix).reshape(batch_size, chunk_frames, full_body_joint_count * 6)
            
            if i > 0 and latent_last is not None:
                # Seed the start of the current input with the end of the previous output
                full_body_target_pose_6d[:, :pre_frames, :] = latent_last[:, -pre_frames:, :]
            
            #----
            # Model forwarding and computing the needed variables for the losses
            model_output = model(full_body_target_pose_6d)
            reconstructed_pose = model_output["rec_pose"].transpose(1, 2) # Transpose is important here to get the right [Batch, frames, channel]
            reconstructed_full_body_pose_6d = reconstructed_pose.reshape(batch_size, chunk_frames, full_body_joint_count, 6)
            reconstructed_full_body_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_full_body_pose_6d)
            reconstructed_full_body_pose_6d = rc.matrix_to_rotation_6d(reconstructed_full_body_pose_matrix)
            reconstructed_pose_axis_angle = rc.matrix_to_axis_angle(reconstructed_full_body_pose_matrix).reshape(batch_size * chunk_frames, -1)
            
            # Store for next iteration's history
            latent_last = reconstructed_full_body_pose_6d.reshape(batch_size, chunk_frames, -1)

            # Create all variables in denormalised space
            if _stage1_model_cache["normalize"]:
                reconstructed_pose = (reconstructed_pose_axis_angle * norm_std) + norm_mean # Denormalised
                reconstructed_pose_axis_angle = reconstructed_pose.reshape(batch_size, chunk_frames, full_body_joint_count, 3)
                reconstructed_full_body_pose_matrix = rc.axis_angle_to_matrix(reconstructed_pose_axis_angle)
                reconstructed_full_body_pose_6d = rc.matrix_to_rotation_6d(reconstructed_full_body_pose_matrix)
            #----

            # Placement into the final buffer
            if i == 0:
                final_reconstructed_6d[:, start_frame:end_frame] = reconstructed_full_body_pose_6d
            else:
                # Only write the new frames
                new_start = start_frame + pre_frames
                final_reconstructed_6d[:, new_start:end_frame] = reconstructed_full_body_pose_6d[:, pre_frames:]
    
    reconstructed_6d = final_reconstructed_6d # [1, num_frames, 43, 6]
    reconstructed_matrix = rc.rotation_6d_to_matrix(reconstructed_6d)
    final_reconstructed_pose = rc.matrix_to_axis_angle(reconstructed_matrix).reshape(batch_size * valid_frames, -1) # [frames, channels]

    reconstructed_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device)
    reconstructed_loader.overwrite_with_reconstructed_pose(final_reconstructed_pose)
    
    output_video_path = utils.generate_reconstruction_comparison_video(original_loader, reconstructed_loader, progress=progress, use_cache=False)
    
    return output_video_path, "Status: Side-by-Side Reconstruction Complete."

def render_full_body_reconstruction_tab(device):
    with gr.TabItem("Stage 1: Reconstruction"):
        gr.Markdown("### 1. Upload Model Files")
        with gr.Row():
            full_config = create_file_browser("Full Body Config (.json)", extensions=[".json"], file_count="single")
            full_body_checkpoint = create_file_browser("Full Body Checkpoint (.bin)", extensions=[".bin"], file_count="single")
        
        load_model_button = gr.Button("Initialize Stage 1 model")
        status_markdown = gr.Markdown("Status: Waiting for model...")

        gr.Markdown("---")
        gr.Markdown("### 2. Compare Original vs Reconstruction")
        with gr.Row():
            target_npz_file = create_file_browser("Target .NPZ file", extensions=[".npz"], file_count="single")
            generate_video_button = gr.Button("Generate Side-by-Side Video", variant="primary")
        
        output_video_component = gr.Video(label="Front View: Original (Left) | Reconstructed (Right)")

        load_model_button.click(
            fn=load_model_from_files,
            inputs=[full_config, full_body_checkpoint, gr.State(device)],
            outputs=status_markdown
        )
        
        generate_video_button.click(
            fn=reconstruct_and_render,
            inputs=[target_npz_file, gr.State(device)],
            outputs=[output_video_component, status_markdown]
        )