import gradio as gr
import torch
import torch.nn.functional as F
import json
import math
import os
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from stage_2_models.autoregressive_transformer import AutoRegressiveMotionTransformer
from training.stage2_trainer_seamless import load_stage_1_checkpoint
from utils import rotation_conversions as rc
import data_processing.utils as utils
from data_processing.seamless_segmenter import SeamlessSegmenter, get_relationship_id
from data_processing.seamless_segmenter import relationship_detail_map
from visualisation.file_browser import create_file_browser
import visualisation.reconstruction_tab_whole_body as stage_1_reconstruction

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

_stage2_runtime_cache = {
    "stage1_model": None,
    "stage2_model": None,
    "max_seq_len": None,
    "code_dim": None,
    "downsample_factor": None,
    "segmenter": None 
}

def load_stage2_and_stage1_pipeline(stage2_config_path, stage2_checkpoint_path, segmenter_config_path, device_str):
    """
    Starts up both the stage 1 and stage 2 model, the stage 1 model is loaded from the stage 2 config.
    """
    device = torch.device(device_str)
    try:
        with open(stage2_config_path, 'r') as f:
            config_dict = json.load(f)
            
        args = SimpleNamespace(**config_dict)
        args.data = SimpleNamespace(**config_dict["data"])
        args.training_params = SimpleNamespace(**config_dict["training_params"])
        args.loss_weights = SimpleNamespace(**config_dict["loss_weights"])
        args.stage_1_pretrain = SimpleNamespace(**config_dict["stage_1_pretrain"])
        args.model_params = SimpleNamespace(**config_dict["model_params"])
        
        # Load stage 1
        m_s1, stage1_model = load_stage_1_checkpoint(
            args.stage_1_pretrain.checkpoint_path,
            args.stage_1_pretrain.config_path,
            device
        )
        
        # Load stage 2
        stage2_model = AutoRegressiveMotionTransformer(args, vocab_size=m_s1["code_dim"] + 1).to(device)
        
        stage2_model.load_state_dict(torch.load(stage2_checkpoint_path, map_location=device))
        stage2_model.eval()
        
        # Additional cache info
        _stage2_runtime_cache["stage1_model"] = stage1_model
        _stage2_runtime_cache["stage2_model"] = stage2_model
        _stage2_runtime_cache["max_seq_len"] = args.model_params.max_seq_len
        _stage2_runtime_cache["code_dim"] = m_s1["code_dim"]
        _stage2_runtime_cache["downsample_factor"] = m_s1.get("downsample_factor", 4)
        
        # Initialise segmenter using the explicitly supplied configuration parameters
        if _stage2_runtime_cache["segmenter"] is None:
            with open(segmenter_config_path, 'r') as f:
                seg_config = json.load(f)
            
            # Clean comments out of the parameter dictionary block
            seg_config = {k: v for k, v in seg_config.items() if "comment" not in k.lower()}
            
            # Prevent the segmenter pipeline from skipping frame evaluations
            seg_config["relationship_detail_mandatory"] = False
            seg_config["personality_detail_mandatory"] = False
            seg_config["filter_out_of_screen_hands"] = False
            
            segmenter = SeamlessSegmenter(dataset=None, **seg_config)
            segmenter.bert_model = segmenter.bert_model.to(device)
            segmenter.hubert_model = segmenter.hubert_model.to(device)
            _stage2_runtime_cache["segmenter"] = segmenter

        return "Status: Stage 1, Stage 2, and Segmenter feature engines successfully active!"
    except Exception as e:
        return f"Status: Framework loading failure - {str(e)}"

def get_loop_variables(device, chunk_frames, num_frames, stride, batch_size, full_body_joint_count):
    num_chunks = math.floor((num_frames - chunk_frames) / stride) + 1
    
    valid_frames = (num_chunks - 1) * stride + chunk_frames
    pre_frames = chunk_frames - stride

    weight_mask = torch.zeros((batch_size, valid_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=pre_frames, device=device).view(1, -1, 1, 1)

    return chunk_frames, stride, num_chunks, full_body_joint_count, batch_size, valid_frames, pre_frames, weight_mask, fade_in, fade_out

def setup_reconstruction(target_npz_path, device):
    if _stage2_runtime_cache["stage1_model"] is None or _stage2_runtime_cache["stage2_model"] is None or _stage2_runtime_cache["segmenter"] is None:
        return None, "Status: Please load the model(s) and segmenter first."

    segmenter = _stage2_runtime_cache["segmenter"]

    original_loader = utils.SeamlessSMPLHLoader(target_npz_path, device=device)
    # Initialize asset dataset and loaders
    original_loader = utils.SeamlessSMPLHLoader(target_npz_path, device=device, with_meta_data=True, load_vertices=True)
    partner_loader = None
    if original_loader.meta_data and original_loader.meta_data.partner_full_id:
        search_root = Path(target_npz_path).parents[2]
        disk_map = {path.stem: str(path.absolute()) for path in search_root.rglob("*.npz")}
        partner_path = disk_map.get(original_loader.meta_data.partner_full_id)
        if partner_path:
            partner_loader = utils.SeamlessSMPLHLoader(partner_path, device=device, with_meta_data=True, load_vertices=False)
    
    # Get the aligned data from the segmenter (still in full shape all frames together)
    segment_data = segmenter._get_loader_segmentation_data(original_loader, partner_loader)
    partner_segment_data = None
    if partner_loader is not None:
        partner_segment_data = segmenter._get_loader_segmentation_data(partner_loader, original_loader)
    else:
        print("\nWARNING: PARTNER WAS NOT FOUND! USING ZERO DATA FOR PARTNER AUDIO\n")
    relationship = get_relationship_id(original_loader)
    
    num_frames = segment_data.aligned_pose.shape[0]
    chunk_frames, stride, num_chunks, full_body_joint_count, batch_size, valid_frames, pre_frames, weight_mask, fade_in, fade_out = get_loop_variables(device, 64, num_frames, 
                                                                                                                       _stage2_runtime_cache["stage1_model"].downsample_factor,
                                                                                                                       1, 43)

    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)
    
    return (original_loader, segment_data, chunk_frames, stride, num_chunks, 
            full_body_joint_count, batch_size, valid_frames, pre_frames, 
            final_reconstructed_6d, weight_mask, fade_in, fade_out, partner_segment_data)

def sample_top_p_k(logits, top_k=50, top_p=0.9, temperature=1.0):
    logits = logits / max(temperature, 1e-5)
    
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate_motion_sequence(device, segment_data, partner_segment_data, chunk_frames, stride, num_chunks, batch_size, temperature=1.0, top_p=0.90, top_k=50):
    downsample_factor =  _stage2_runtime_cache["stage1_model"].downsample_factor
    code_dim = _stage2_runtime_cache["stage1_model"].categories
    final_amount_of_tokens = segment_data.aligned_pose.shape[0] // downsample_factor
    generated_tokens = torch.full((batch_size, final_amount_of_tokens, downsample_factor), code_dim, dtype=torch.long, device=device) # [1, 16, 4] for example
    chunk_codes = (chunk_frames // downsample_factor)
    # Prepare the features
    segment_data.aligned_wav2vec_audio = torch.from_numpy(segment_data.aligned_wav2vec_audio).float().to(device)
    segment_data.personality_vector = torch.from_numpy(segment_data.personality_vector).float().to(device)
    personality_input = segment_data.personality_vector.unsqueeze(0)
    #
    with torch.no_grad():
        model = _stage2_runtime_cache["stage2_model"]
        generated_tokens[:, 0, :] = torch.randint(0, code_dim, (batch_size, 1, 4), device=device) # Set the first token to a random seed token
        # Generate the initial 15 frames for the first chunk
        for j in range(1, chunk_codes):
            audio_input = segment_data.aligned_wav2vec_audio[0:chunk_frames].unsqueeze(0)
            # Slice the 15-token input context window (unpredicted tokens remain masked as code_dim)
            tokens_input = generated_tokens[:, 0:chunk_codes - 1, :]
            logits, _ = model(
                motion_tokens=tokens_input,
                audio=audio_input,
                partner_audio=None, bert=None, emotions=None, social=None, conversation_state=None,
                personality=personality_input,
                cond_drop_prob=0.0
            )
            
            # Grab logits at index j-1 to predict the token at index j
            next_frame_logits = logits[:, j - 1, :, :]
            
            next_frame_tokens = []
            for layer_idx in range(4):
                layer_logits = next_frame_logits[:, :, layer_idx]
                sampled_token = torch.argmax(layer_logits, dim=-1, keepdim=True)
                next_frame_tokens.append(sampled_token)
                
            next_frame_tensor = torch.stack(next_frame_tokens, dim=-1).squeeze(1)
            generated_tokens[:, j, :] = next_frame_tensor

        # Run the autoregressive loop
        for i in range(num_chunks):
            start_frame = i * stride
            end_frame = start_frame + chunk_frames

            # Slice exactly 15 tokens for the input window to match training shapes
            tokens_input = generated_tokens[:, start_frame//downsample_factor : (end_frame//downsample_factor) - 1, :]
            audio_input = segment_data.aligned_wav2vec_audio[start_frame:end_frame].unsqueeze(0) # Unsqueeze to add batch dimension

            logits, _ = model(
                motion_tokens=tokens_input,
                audio=audio_input,
                partner_audio=None,
                bert=None, emotions=None, social=None, conversation_state=None,
                personality=personality_input,
                cond_drop_prob=0.0
            )

            next_frame_logits = logits[:, -1, :, :] # Grab the just generated motion code (last one in the sequence)
            
            next_frame_tokens = []
            # Sample each codebook layer hierarchically for this frame
            for layer_idx in range(4):
                layer_logits = next_frame_logits[:, :, layer_idx] # Shape: [B, vocab_size]
                
                sampled_token = torch.argmax(layer_logits, dim=-1, keepdim=True)
                # sampled_token = sample_top_p_k(layer_logits, top_p=top_p, temperature=temperature) # Shape: [B, 1]
                next_frame_tokens.append(sampled_token)
                
            next_frame_tensor = torch.stack(next_frame_tokens, dim=-1).squeeze(1) # Shape: [B, 4]
            
            # Assign the brand new frame directly into the pre-allocated token history
            target_idx = (end_frame // downsample_factor) - 1
            if target_idx < final_amount_of_tokens:
                generated_tokens[:, target_idx, :] = next_frame_tensor
    return generated_tokens

def reconstruct_and_render(target_npz_file, device, top_k, top_p, temperature, progress=gr.Progress()):
    setup_data = setup_reconstruction(target_npz_file, device)
    if isinstance(setup_data, tuple) and len(setup_data) == 2: # for if the model is not loaded
        return setup_data
        
    (original_loader, segment_data, chunk_frames, stride, num_chunks, 
        full_body_joint_count, batch_size, valid_frames, pre_frames, 
        final_reconstructed_6d, weight_mask, fade_in, fade_out, partner_segment_data) = setup_data # unpack setup_data

    pose_codes = generate_motion_sequence(
        device, 
        segment_data, partner_segment_data, 
        chunk_frames, stride, num_chunks, batch_size, 
        temperature=temperature, top_p=top_p, top_k=top_k
    )

    # Update the variables since we use a different stride now
    num_frames = segment_data.aligned_pose.shape[0]
    chunk_frames, stride, num_chunks, full_body_joint_count, batch_size, valid_frames, pre_frames, weight_mask, fade_in, fade_out = get_loop_variables(device, 64, num_frames, 
                                                                                                                       20,
                                                                                                                       1, 43)
    # Turn the large tensor of motion codes into a list of tensors of size "16" to match stage 1 reconstruction
    downsample_factor =  _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_codes = (chunk_frames // downsample_factor)
    pose_codes_list = list(pose_codes.view(1, -1, chunk_codes, downsample_factor).unbind(dim=1))
    print("="*50)
    print(f"Num chunks is of size {num_chunks} and the list is of size {pose_codes_list} and pose_codes has a shape of {pose_codes.shape}")
    print("="*50)
    final_6d_pose = stage_1_reconstruction.decode_sequence(
        device, pose_codes_list, chunk_frames, stride, num_chunks, 
        full_body_joint_count, batch_size, pre_frames, 
        final_reconstructed_6d, weight_mask, fade_in, fade_out, model=_stage2_runtime_cache["stage1_model"]
    )

    return stage_1_reconstruction.visualize_reconstruction(
        target_npz_file, device, batch_size, valid_frames, 
        final_6d_pose, original_loader, progress
    )

def render_stage2_generation_tab(device):
    with gr.TabItem("Stage 2: Audio Gesture Generation"):
        gr.Markdown("### 1. Load System Framework (Stage 2 + Stage 1 Dependency)")
        with gr.Row():
            stage2_config = create_file_browser("Stage 2 Config Path (.json)", extensions=[".json"], file_count="single", default_value="/data/scratch/davwer_projects/mpi-thesis/configs/stage2_model.json")
            stage2_checkpoint = create_file_browser("Stage 2 Checkpoint Path (.bin)", extensions=[".bin"], file_count="single", default_value="/data/scratch/davwer_projects/mpi-thesis/checkpoints_stage_2/stage_2_model_ce_mse_soft/20260609_2034/stage2_final_epoch_199.bin")
            segmenter_config = create_file_browser("Segmenter Config Path (.json)", extensions=[".json"], file_count="single", default_value="/data/scratch/davwer_projects/mpi-thesis/configs/seamless_segmenter_stage_2.json")
            
        load_pipeline_btn = gr.Button("Initialize Integrated Model Pipeline", variant="secondary")
        pipeline_status = gr.Markdown("Status: Waiting for initialization parameters...")
        
        gr.Markdown("---")
        gr.Markdown("### 2. Autoregressive Sampling Generation Parameters")
        with gr.Row():
            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.05, label="Temperature")
            top_k = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top-K")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.90, step=0.01, label="Top-P (Nucleus)")
            
        gr.Markdown("### 3. Select Target Conversation Interaction File")
        with gr.Row():
            target_npz_file = create_file_browser("Target .NPZ Sequence File Path", extensions=[".npz"], file_count="single", default_value="data/seamless/downloads/train_first_1000_imitator_samples/naturalistic/train/0001/0022/V00_S0096_I00000126_P0138.npz")
            generate_gesture_btn = gr.Button("Generate Gestures from Audio Context", variant="primary")
            
        output_video_component = gr.Video(label="Front View Evaluation: Ground Truth (Left) | Generated from Audio (Right)")

        load_pipeline_btn.click(
            fn=load_stage2_and_stage1_pipeline,
            inputs=[stage2_config, stage2_checkpoint, segmenter_config, gr.State(device)],
            outputs=pipeline_status
        )
        
        generate_gesture_btn.click(
            fn=reconstruct_and_render,
            inputs=[target_npz_file, gr.State(device), top_k, top_p, temperature],
            outputs=[output_video_component, pipeline_status]
        )