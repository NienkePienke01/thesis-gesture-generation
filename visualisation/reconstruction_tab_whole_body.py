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
from models_dyadit.vqvae_downsampled import TemporalVertexCodecDownsampled
import data_processing.utils as utils
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import math

_stage1_model_cache = {
    "full_body_model": None,
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
        m = full_config_dict["model_params"]
        model = TemporalVertexCodecDownsampled(
            n_vertices=m["nb_joints"],
            latent_dim=m["output_emb_width"],
            categories=m["code_dim"],
            residual_depth=m["depth"],
            downsample_factor=m["downsample_factor"],
        ).to(device)

        model.load_state_dict(torch.load(full_body_checkpoint, map_location=device))
        model.eval()

        _stage1_model_cache["full_body_model"] = model
        
        return "Status: Full Body model loaded successfully."
    except Exception as e:
        return f"Status: Error loading model - {str(e)}"

def setup_reconstruction(target_npz_file, device):
    if _stage1_model_cache["full_body_model"] is None:
        return None, "Status: Please load the model first."

    original_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device)
    full_pose_sequence = original_loader.get_trainable_pose().to(device)
    
    chunk_frames = 64
    stride = 20
    num_frames = full_pose_sequence.shape[0]
    num_chunks = math.floor((num_frames - chunk_frames) / stride) + 1
    full_body_joint_count = 43
    batch_size = 1
    
    valid_frames = (num_chunks - 1) * stride + chunk_frames
    pre_frames = chunk_frames - stride

    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)
    weight_mask = torch.zeros((batch_size, valid_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    
    return (original_loader, full_pose_sequence, chunk_frames, stride, num_chunks, 
            full_body_joint_count, batch_size, valid_frames, pre_frames, 
            final_reconstructed_6d, weight_mask, fade_in, fade_out)

def encode_sequence(full_pose_sequence, chunk_frames, stride, num_chunks, full_body_joint_count, batch_size):
    encoded_poses = []
    with torch.no_grad():
        model = _stage1_model_cache["full_body_model"]
        for i in range(num_chunks):
            start_frame = i * stride
            end_frame = start_frame + chunk_frames
            chunk_tensor = full_pose_sequence[start_frame:end_frame].unsqueeze(0)
            
            full_body_target_pose = chunk_tensor
            full_body_target_pose_matrix = rc.axis_angle_to_matrix(full_body_target_pose.reshape(batch_size, chunk_frames, full_body_joint_count, 3))
            full_body_target_pose_6d = rc.matrix_to_rotation_6d(full_body_target_pose_matrix).reshape(batch_size, chunk_frames, full_body_joint_count * 6)
            
            encoded_pose = model.encode(full_body_target_pose_6d)
            encoded_poses.append(encoded_pose)
            
    return encoded_poses

def decode_sequence(device, encoded_poses, chunk_frames, stride, num_chunks, full_body_joint_count, batch_size, pre_frames, final_reconstructed_6d, weight_mask, fade_in, fade_out, model=None):
    with torch.no_grad():
        if model is None:
            model = _stage1_model_cache["full_body_model"]
        for i in range(num_chunks):
            start_frame = i * stride
            end_frame = start_frame + chunk_frames
            
            encoded_pose = encoded_poses[i]
            reconstructed_pose = model.decode(encoded_pose)

            reconstructed_full_body_pose_6d = reconstructed_pose.reshape(batch_size, chunk_frames, full_body_joint_count, 6)
            reconstructed_full_body_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_full_body_pose_6d)
            reconstructed_full_body_pose_6d = rc.matrix_to_rotation_6d(reconstructed_full_body_pose_matrix)

            chunk_weight = torch.ones((batch_size, chunk_frames, full_body_joint_count, 6), device=device)
            if i > 0:
                chunk_weight[:, :pre_frames, :, :] *= fade_in
            if i < num_chunks-1:
                chunk_weight[:, -pre_frames:, :, :] *= fade_out

            final_reconstructed_6d[:, start_frame:end_frame] += reconstructed_full_body_pose_6d * chunk_weight
            weight_mask[:, start_frame:end_frame] += chunk_weight[:, :, :1, :1]
        
        final_reconstructed_6d = final_reconstructed_6d / (weight_mask + 1e-8)
            
    return final_reconstructed_6d

def visualize_reconstruction(target_npz_file, device, batch_size, valid_frames, final_reconstructed_6d, original_loader, progress, include_audio=False):
    reconstructed_6d = final_reconstructed_6d
    reconstructed_matrix = rc.rotation_6d_to_matrix(reconstructed_6d)
    final_reconstructed_pose = rc.matrix_to_axis_angle(reconstructed_matrix).reshape(batch_size * valid_frames, -1)

    reconstructed_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device)
    reconstructed_loader.overwrite_with_reconstructed_pose(final_reconstructed_pose)
    
    output_video_path = utils.generate_reconstruction_comparison_video(original_loader, reconstructed_loader, progress=progress, use_cache=False, include_audio=include_audio)
    
    return output_video_path, "Status: Side-by-Side Reconstruction Complete."

def reconstruct_and_render(target_npz_file, device, progress=gr.Progress()):
    setup_data = setup_reconstruction(target_npz_file, device)
    if isinstance(setup_data, tuple) and len(setup_data) == 2: # for if the model is not loaded
        return setup_data
        
    (original_loader, full_pose_sequence, chunk_frames, stride, num_chunks, 
     full_body_joint_count, batch_size, valid_frames, pre_frames, 
     final_reconstructed_6d, weight_mask, fade_in, fade_out) = setup_data # unpack setup_data

    encoded_poses = encode_sequence(
        full_pose_sequence, chunk_frames, stride, num_chunks, 
        full_body_joint_count, batch_size
    )

    final_reconstructed_6d = decode_sequence(
        device, encoded_poses, chunk_frames, stride, num_chunks, 
        full_body_joint_count, batch_size, pre_frames, 
        final_reconstructed_6d, weight_mask, fade_in, fade_out
    )

    return visualize_reconstruction(
        target_npz_file, device, batch_size, valid_frames, 
        final_reconstructed_6d, original_loader, progress
    )

def render_full_body_reconstruction_tab(device):
    with gr.TabItem("Stage 1: Reconstruction"):
        gr.Markdown("### 1. Upload Model Files")
        with gr.Row():
            full_config = create_file_browser("Full Body Config (.json)", extensions=[".json"], file_count="single", default_value="configs/stage1_new_dyadit_model_attempt.json")
            full_body_checkpoint = create_file_browser("Full Body Checkpoint (.bin)", extensions=[".bin"], file_count="single", default_value="checkpoints/stage1_new_dyadit_model_attempt/20260519_0207/model_final_epoch_799_val_loss_0.03552431985735893.bin")
        
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