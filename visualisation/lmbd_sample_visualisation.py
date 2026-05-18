import gradio as gr
import torch
import numpy as np
import os
import pickle
import lmdb
import math
from pathlib import Path
from utils import rotation_conversions as rc
import data_processing.utils as utils
from visualisation.file_browser import create_file_browser

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def fetch_ordered_lmdb_chunks(source_path, file_id):
    """
    Opens the LMDB database environment, scans for entries belonging to the 
    specified interaction file_id, extracts the raw poses, and orders them sequentially.
    """
    lmdb_list = []
    src_env = lmdb.open(str(source_path), readonly=True, lock=False)
    
    with src_env.begin() as src_txn:
        file_groups = {}
        cursor = src_txn.cursor()
        
        # Group entries by their seamless_id
        for key, value in cursor:
            idx = int(key.decode('ascii'))
            entry = pickle.loads(value)
            f_id = entry["seamless_id"]
            
            if f_id not in file_groups:
                file_groups[f_id] = []
            file_groups[f_id].append(idx)
        
        if file_id not in file_groups:
            src_env.close()
            raise ValueError(f"Interaction ID '{file_id}' not found in this LMDB environment.")
            
        # Sort indices to guarantee timeline continuity
        indices = file_groups[file_id]
        indices.sort() 
        
        # Retrieve the sorted chunk tensors
        for idx in indices:
            raw_data = src_txn.get(f"{idx:08d}".encode("ascii"))
            entry = pickle.loads(raw_data)
            pose_tensor = torch.tensor(entry["pose"], dtype=torch.float32) # Expected shape: [64, 129]
            lmdb_list.append(pose_tensor)
            
    src_env.close()
    return lmdb_list

def reconstruct_and_render_lmdb(lmdb_list, target_npz_file, device, progress=gr.Progress()):
    original_loader = utils.SeamlessSMPLHLoader(target_npz_file, device=device)
    
    chunk_frames = 64
    stride = 20

    num_chunks = len(lmdb_list)
    full_body_joint_count = 43
    batch_size = 1 # We do not batch here, so its just hard set to 1
    #
    valid_frames = (num_chunks - 1) * stride + chunk_frames
    latent_last = None
    pre_frames = chunk_frames - stride
    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)
    
    with torch.no_grad():
        for i in range(num_chunks):
            start_frame = i * stride
            end_frame = start_frame + chunk_frames
            chunk_tensor = lmdb_list[i]
            
            # Full body
            full_body_target_pose = chunk_tensor
            full_body_target_pose_matrix = rc.axis_angle_to_matrix(full_body_target_pose.reshape(batch_size, chunk_frames, full_body_joint_count, 3))
            full_body_target_pose_6d = rc.matrix_to_rotation_6d(full_body_target_pose_matrix).reshape(batch_size, chunk_frames, full_body_joint_count * 6)
            
            if i > 0 and latent_last is not None:
                # Seed the start of the current input with the end of the previous output
                full_body_target_pose_6d[:, :pre_frames, :] = latent_last[:, -pre_frames:, :]

            # Normally model output is here
            reconstructed_pose = full_body_target_pose_6d
            #
            # reconstructed_full_body_pose_6d = reconstructed_pose.transpose(1, 2).reshape(batch_size, chunk_frames, full_body_joint_count, 6)
            reconstructed_full_body_pose_6d = reconstructed_pose.reshape(batch_size, chunk_frames, full_body_joint_count, 6) # Debugging
            reconstructed_full_body_pose_matrix = rc.rotation_6d_to_matrix(reconstructed_full_body_pose_6d)
            reconstructed_full_body_pose_6d = rc.matrix_to_rotation_6d(reconstructed_full_body_pose_matrix)
            # Store for next iteration's history
            latent_last = reconstructed_full_body_pose_6d.reshape(batch_size, chunk_frames, -1)

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
    
    output_video_path = utils.generate_reconstruction_comparison_video(original_loader, reconstructed_loader, progress=progress)
    
    return output_video_path, "Status: Side-by-Side Reconstruction Complete."

def process_lmdb_visualisation(target_npz_file, lmdb_folder, device):
    """
    Pulls the interaction ID from the file name stem, strings the binary database
    sequences together, and outputs the comparison render.
    """
    if not target_npz_file or not lmdb_folder:
        return None, "Status: Please select both an NPZ reference file and an LMDB directory."
    try:
        # Extract the base file name to serve as our interaction identifier
        file_id = Path(target_npz_file).stem
        
        # Load and sort data slices
        lmdb_list = fetch_ordered_lmdb_chunks(lmdb_folder, file_id)
        
        # Process and compile rendering
        video_path, status_msg = reconstruct_and_render_lmdb(lmdb_list, target_npz_file, device)
        return video_path, status_msg
    except Exception as e:
        return None, f"Status: Error processing LMDB structures - {str(e)}"

def render_lmdb_visualiser_tab(device):
    """
    Renders a dedicated, independent Gradio workspace interface tab.
    """
    with gr.TabItem("Data Pipeline Diagnostic: LMDB Visualiser"):
        gr.Markdown("### 1. Match Raw LMDB Slices Against Target Sequence")
        gr.Markdown("This panel extracts chunk frames from your database folder using the selected `.npz` filename as the search ID. It stitches them sequentially to isolate and catch data pipeline bugs.")
        
        with gr.Row():
            target_npz_file = create_file_browser("Target Reference .NPZ File", extensions=[".npz"], file_count="single")
            lmdb_folder_path = create_file_browser("Source LMDB Database Directory", file_count="folder")
        
        generate_video_button = gr.Button("Restitch & Render Database Chunks", variant="primary")
        status_markdown = gr.Markdown("Status: Waiting for input vectors...")
        
        output_video_component = gr.Video(
            label="Stitch Diagnostic View: Ground-Truth Raw File (Left) | Pipeline Restitched Chunks (Right)"
        )

        generate_video_button.click(
            fn=process_lmdb_visualisation,
            inputs=[target_npz_file, lmdb_folder_path, gr.State(device)],
            outputs=[output_video_component, status_markdown]
        )