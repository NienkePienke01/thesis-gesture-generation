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

_stage1_model_cache = {
    "hand_model": None, 
    "upper_body_model": None
}

def load_models_from_files(hand_config, upper_body_config, hand_checkpoint, upper_body_checkpoint, device):
    """
    Initializes models using the four explicitly uploaded files.
    """
    try:
        # Initialize Hand Model
        with open(hand_config, 'r') as f:
            hand_config_dict = json.load(f)
        hand_args = SimpleNamespace(**hand_config_dict)
        hand_model = RVQVAE(hand_args, **hand_config_dict["model_params"]).to(device)
        hand_model.load_state_dict(torch.load(hand_checkpoint, map_location=device, weights_only=True))
        hand_model.eval()

        # Initialize Upper Body Model
        with open(upper_body_config, 'r') as f:
            upper_body_config_dict = json.load(f)
        upper_body_args = SimpleNamespace(**upper_body_config_dict)
        upper_body_model = RVQVAE(upper_body_args, **upper_body_config_dict["model_params"]).to(device)
        upper_body_model.load_state_dict(torch.load(upper_body_checkpoint, map_location=device, weights_only=True))
        upper_body_model.eval()

        _stage1_model_cache["hand_model"] = hand_model
        _stage1_model_cache["upper_body_model"] = upper_body_model
        
        return "Status: Hand and Upper Body models loaded successfully."
    except Exception as e:
        return f"Status: Error loading models - {str(e)}"

def reconstruct_and_render(target_npz_file, device, progress=gr.Progress()):
    if _stage1_model_cache["hand_model"] is None or _stage1_model_cache["upper_body_model"] is None:
        return None, "Status: Please load both models first."

    original_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device)
    full_pose_sequence = original_loader.get_trainable_pose().to(device) # Shape: [Total_Frames, 129]
    
    chunk_frames = 64
    num_frames = full_pose_sequence.shape[0]
    num_chunks = num_frames // chunk_frames
    reconstructed_pose_list = []
    upper_body_joint_count = 13
    hands_joint_count = 30
    batch_size = 1 # We do not batch here, so its just hard set to 1
    
    # For debugging
    num_chunks = min([num_chunks, 5])
    #
    with torch.no_grad():
        for i in range(num_chunks):
            start_frame = i * chunk_frames
            end_frame = (i + 1) * chunk_frames
            chunk_tensor = full_pose_sequence[start_frame:end_frame].unsqueeze(0) # Shape: [1, chunk_frames, 129]
            
            # Upper body
            upper_body_target_pose = chunk_tensor[:, :, :39]
            upper_body_target_pose_matrix = rc.axis_angle_to_matrix(upper_body_target_pose.reshape(batch_size, chunk_frames, upper_body_joint_count, 3))
            upper_body_target_pose_6d = rc.matrix_to_rotation_6d(upper_body_target_pose_matrix).reshape(batch_size, chunk_frames, upper_body_joint_count * 6)
            #Debug
            raw_output = _stage1_model_cache["upper_body_model"](upper_body_target_pose_6d)["rec_pose"]
            movement_variance = torch.var(raw_output, dim=1).mean().item()
            reconstructed_upper_body_pose_6d = raw_output.reshape(batch_size, chunk_frames, upper_body_joint_count, 6)
            #
            # reconstructed_upper_body_pose_6d = _stage1_model_cache["upper_body_model"](upper_body_target_pose_6d)["rec_pose"].reshape(batch_size, chunk_frames, upper_body_joint_count, 6)
            reconstructed_upper_body_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_upper_body_pose_6d)
            reconstructed_upper_body_pose_axis_angles = rc.matrix_to_axis_angle(reconstructed_upper_body_pose_matrix).reshape(batch_size * chunk_frames, -1)

            # Hands
            hands_target_pose = chunk_tensor[:, :, 39:129]
            hands_target_pose_matrix = rc.axis_angle_to_matrix(hands_target_pose.reshape(batch_size, chunk_frames, hands_joint_count, 3))
            hands_target_pose_6d = rc.matrix_to_rotation_6d(hands_target_pose_matrix).reshape(batch_size, chunk_frames, hands_joint_count * 6)
            reconstructed_hands_pose_6d = _stage1_model_cache["hand_model"](hands_target_pose_6d)["rec_pose"].reshape(batch_size, chunk_frames, hands_joint_count, 6)
            reconstructed_hands_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_hands_pose_6d)
            reconstructed_hands_pose_axis_angles = rc.matrix_to_axis_angle(reconstructed_hands_pose_matrix).reshape(batch_size * chunk_frames, -1)

            combined_reconstruction = torch.cat([reconstructed_upper_body_pose_axis_angles, reconstructed_hands_pose_axis_angles], dim=-1)
            reconstructed_pose_list.append(combined_reconstruction.squeeze(0))

            # Debug
            if i == 1:
                print("---------DEBUG---------")
                print(f"Movement Variance: {movement_variance:.8f}")
                print(f"Upper Body 6D Input Shape: {upper_body_target_pose_6d.shape}")
                print(f"Reshaped 6D Pose Shape: {reconstructed_upper_body_pose_6d.shape}")
                print(f"Max Axis Angle value: {reconstructed_upper_body_pose_axis_angles.abs().max().item():.2f}")
    
    if not reconstructed_pose_list:
        return None, "Status: Input sequence too short for chunk_frames chunks based on the given config."

    final_reconstructed_pose = torch.cat(reconstructed_pose_list, dim=0)
    reconstructed_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device)
    reconstructed_loader.overwrite_with_reconstructed_pose(final_reconstructed_pose)

    output_video_path = utils.generate_reconstruction_comparison_video(original_loader, reconstructed_loader, progress=progress)
    
    return output_video_path, "Status: Side-by-Side Reconstruction Complete."

def render_reconstruction_tab(device):
    with gr.TabItem("Stage 1: Reconstruction"):
        gr.Markdown("### 1. Upload Model Files")
        with gr.Row():
            with gr.Column():
                hand_config = create_file_browser("Hand Config (.json)", extensions=[".json"], file_count="single")
                hand_checkpoint = create_file_browser("Hand Checkpoint (.bin)", extensions=[".bin"], file_count="single")
            with gr.Column():
                upper_body_config = create_file_browser("Upper Body Config (.json)", extensions=[".json"], file_count="single")
                upper_body_checkpoint = create_file_browser("Upper Body Checkpoint (.bin)", extensions=[".bin"], file_count="single")
        
        load_models_button = gr.Button("Initialize Stage 1 Models")
        status_markdown = gr.Markdown("Status: Waiting for models...")

        gr.Markdown("---")
        gr.Markdown("### 2. Compare Original vs Reconstruction")
        with gr.Row():
            target_npz_file = create_file_browser("Target .NPZ file", extensions=[".npz"], file_count="single")
            generate_video_button = gr.Button("Generate Side-by-Side Video", variant="primary")
        
        output_video_component = gr.Video(label="Front View: Original (Left) | Reconstructed (Right)")

        load_models_button.click(
            fn=load_models_from_files,
            inputs=[hand_config, upper_body_config, hand_checkpoint, upper_body_checkpoint, gr.State(device)],
            outputs=status_markdown
        )
        
        generate_video_button.click(
            fn=reconstruct_and_render,
            inputs=[target_npz_file, gr.State(device)],
            outputs=[output_video_component, status_markdown]
        )