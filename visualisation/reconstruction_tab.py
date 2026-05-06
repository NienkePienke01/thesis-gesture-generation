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

from models.rvq import RVQVAE
import data_processing.utils as utils

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
        with open(hand_config.name, 'r') as f:
            hand_config_dict = json.load(f)
        hand_args = SimpleNamespace(**hand_config_dict)
        hand_model = RVQVAE(hand_args, **hand_config_dict["model_params"]).to(device)
        hand_model.load_state_dict(torch.load(hand_checkpoint.name, map_location=device))
        hand_model.eval()

        # Initialize Upper Body Model
        with open(upper_body_config.name, 'r') as f:
            upper_body_config_dict = json.load(f)
        upper_body_args = SimpleNamespace(**upper_body_config_dict)
        upper_body_model = RVQVAE(upper_body_args, **upper_body_config_dict["model_params"]).to(device)
        upper_body_model.load_state_dict(torch.load(upper_body_checkpoint.name, map_location=device))
        upper_body_model.eval()

        _stage1_model_cache["hand_model"] = hand_model
        _stage1_model_cache["upper_body_model"] = upper_body_model
        
        return "Status: Hand and Upper Body models loaded successfully."
    except Exception as e:
        return f"Status: Error loading models - {str(e)}"

def reconstruct_and_render(target_npz_file, device, progress=gr.Progress()):
    if _stage1_model_cache["hand_model"] is None or _stage1_model_cache["upper_body_model"] is None:
        return None, "Status: Please load both models first."

    # SeamlessSMPLHLoader automatically handles the 129-dim extraction
    original_loader = utils.SeamlessSMPLHLoader(target_npz_file.name, device=device)
    full_pose_sequence = original_loader.get_trainable_pose().to(device) # Shape: [Total_Frames, 129]
    
    # Segment into 64-frame chunks for the RVQVAE
    num_frames = full_pose_sequence.shape[0]
    num_chunks = num_frames // 64
    reconstructed_pose_list = []

    with torch.no_grad():
        for i in range(num_chunks):
            start_frame = i * 64
            end_frame = (i + 1) * 64
            chunk_tensor = full_pose_sequence[start_frame:end_frame].unsqueeze(0) # Shape: [1, 64, 129]
            
            # Split: Upper Body (indices 0-39), Hands (indices 39-129)
            upper_body_input = chunk_tensor[:, :, :39]
            hand_input = chunk_tensor[:, :, 39:]
            
            reconstructed_upper_body = _stage1_model_cache["upper_body_model"](upper_body_input)["rec_pose"]
            reconstructed_hand = _stage1_model_cache["hand_model"](hand_input)["rec_pose"]
            
            # Stitch reconstructed components back together
            combined_reconstruction = torch.cat([reconstructed_upper_body, reconstructed_hand], dim=-1)
            reconstructed_pose_list.append(combined_reconstruction.squeeze(0))
    
    # Concatenate all chunks back into a continuous sequence
    final_reconstructed_pose = torch.cat(reconstructed_pose_list, dim=0)
    reconstructed_loader = copy.deepcopy(original_loader)
    reconstructed_loader.overwrite_with_reconstructed_pose(final_reconstructed_pose)

    output_video_path = utils.generate_reconstruction_comparison_video(original_loader, reconstructed_loader, progress=progress)
    
    return output_video_path, "Status: Side-by-Side Reconstruction Complete."

def render_reconstruction_tab(device):
    with gr.TabItem("Stage 1: Reconstruction"):
        gr.Markdown("### 1. Upload Model Files")
        with gr.Row():
            with gr.Column():
                hand_config = gr.File(label="Hand Config (.json)")
                hand_checkpoint = gr.File(label="Hand Checkpoint (.bin)")
            with gr.Column():
                upper_body_config = gr.File(label="Upper Body Config (.json)")
                upper_body_checkpoint = gr.File(label="Upper Body Checkpoint (.bin)")
        
        load_models_button = gr.Button("Initialize Stage 1 Models")
        status_markdown = gr.Markdown("Status: Waiting for models...")

        gr.Markdown("---")
        gr.Markdown("### 2. Compare Original vs Reconstruction")
        with gr.Row():
            target_npz_file = gr.File(label="Target NPZ File")
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