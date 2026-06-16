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

    return original_loader, segment_data, partner_segment_data, relationship

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

def generate_motion_sequence(device, segment_data, partner_segment_data, chunk_frames, stride, num_chunks, batch_size, temperature=1.0, top_p=0.90, top_k=50, warm_start=False):
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
        if warm_start:
            # Grab the raw pose for the first 64 frames
            gt_pose_chunk = segment_data.aligned_pose[0:chunk_frames].unsqueeze(0).to(device) # Shape: [1, 64, 129]
            
            # Reshape and convert to 6D Rotations (43 joints * 6)
            batch_size = gt_pose_chunk.shape[0]
            full_body_joint_count = 43
            
            gt_pose_matrix = rc.axis_angle_to_matrix(gt_pose_chunk.reshape(batch_size, chunk_frames, full_body_joint_count, 3))
            gt_pose_6d = rc.matrix_to_rotation_6d(gt_pose_matrix).reshape(batch_size, chunk_frames, full_body_joint_count * 6)
            
            # Encode using Stage 1
            stage1_model = _stage2_runtime_cache["stage1_model"]
            encoded_tokens = stage1_model.encode(gt_pose_6d) # Shape: [B, chunk_codes, 4]
            
            # Inject the ground truth tokens into the history
            generated_tokens[:, 0:chunk_codes, :] = encoded_tokens
        else:
            for j in range(1, chunk_codes):
                audio_input = segment_data.aligned_wav2vec_audio[0:chunk_frames].unsqueeze(0)
                
                # Slice the 15-token input context window (unpredicted tokens remain masked as code_dim)
                tokens_input = generated_tokens[:, 0:chunk_codes - 1, :]
                
                # layer 0 pass
                base_logits, _, features, cross_attention_condition = model(
                    motion_tokens=tokens_input,
                    audio=audio_input,
                    partner_audio=None, bert=None, emotions=None, social=None, conversation_state=None,
                    personality=personality_input,
                    cond_drop_prob=0.0,
                    skip_residuals=True
                )
                
                # Grab base logits at index j-1 to predict the token at index j
                current_frame_layer0_logits = base_logits[:, j - 1, :]
                current_frame_layer0_logits[..., code_dim] = float('-inf')  # block the mask token
                sampled_token = sample_top_p_k(current_frame_layer0_logits, top_p=top_p, temperature=temperature, top_k=top_k) # Shape: [B, 1]
                # Embed the Layer 0 token
                sampled_emb = model.motion_embedding(sampled_token) # Shape: [B, 1, d_model]
                
                # Predict Residuals!
                residual_outputs = list(model.residual_predictor(
                    features[:, j - 1 : j, :]
                    cross_attention_condition[:, j - 1 : j, :],
                    sampled_emb
                ))
                
                # Extract and sample the residuals in a greedy fashion
                residual_logits = residual_outputs[model.residual_predictor.n_tokens:]
                sampled_residuals = [torch.argmax(residual_logits[i], dim=-1) for i in range(3)]

                # Combine all 4 layers and write them to the history
                next_frame_tensor = torch.cat([sampled_token] + sampled_residuals, dim=-1) # Shape: [B, 4]
                generated_tokens[:, j, :] = next_frame_tensor

        # Run the autoregressive loop
        for i in range(num_chunks):
            start_frame = i * stride
            end_frame = start_frame + chunk_frames

            # Slice exactly 15 tokens for the input window to match training shapes
            start_token = start_frame // downsample_factor
            end_token = start_token + chunk_codes - 1
            tokens_input = generated_tokens[:, start_token:end_token, :]
            audio_input = segment_data.aligned_wav2vec_audio[start_frame:end_frame].unsqueeze(0) # Unsqueeze to add batch dimension

            base_logits, _, features, cross_attention_condition = model(
                motion_tokens=tokens_input,
                audio=audio_input,
                partner_audio=None,
                bert=None, emotions=None, social=None, conversation_state=None,
                personality=personality_input,
                cond_drop_prob=0.0,
                skip_residuals=True
            )

            current_frame_layer0_logits = base_logits[:, -1, :] 
            current_frame_layer0_logits[..., code_dim] = float('-inf') # Block the mask token from being chosen during nucleus sampling
            sampled_token = sample_top_p_k(current_frame_layer0_logits, top_p=top_p, temperature=temperature, top_k=top_k) # Shape: [B, 1], sample layer 0 token
            sampled_emb = model.motion_embedding(sampled_token) # Shape: [B, 1, d_model], embed the layer 0 token

            # Get layer 1-2-3 logits
            residual_outputs = list(model.residual_predictor(features[:, -1:, :], cross_attention_condition[:, -1:, :], sampled_emb)) # also slice down the context to only this frame

            # Extract the residual logits
            residual_logits = residual_outputs[model.residual_predictor.n_tokens:]

            # Sample the residuals in a greedy fasion
            sampled_residuals = [torch.argmax(residual_logits[i], dim=-1) for i in range(3)]
            next_frame_tensor = torch.cat([sampled_token] + sampled_residuals, dim=-1) # Shape: [B, 4]
            
            # Assign the brand new frame directly into the pre-allocated token history
            target_idx = (end_frame // downsample_factor) - 1
            if target_idx < final_amount_of_tokens:
                generated_tokens[:, target_idx, :] = next_frame_tensor
    return generated_tokens

def reconstruct_and_render(target_npz_file, device, top_k, top_p, temperature, warm_start, progress=gr.Progress()):
    setup_data = setup_reconstruction(target_npz_file, device)
    if isinstance(setup_data, tuple) and len(setup_data) == 2: # for if the model is not loaded
        return setup_data

    original_loader, segment_data, partner_segment_data, relationship = setup_data
    stride = _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_frames =  _stage2_runtime_cache["segmenter"].pose_length
    num_frames = segment_data.aligned_pose.shape[0]
    num_chunks = math.floor((num_frames - chunk_frames) / stride) + 1
    batch_size = 1

    # Do inference on stage 2
    pose_codes = generate_motion_sequence(
        device, 
        segment_data, partner_segment_data, 
        chunk_frames, stride, num_chunks, batch_size, 
        temperature=temperature, top_p=top_p, top_k=top_k, warm_start=warm_start
    )

    # Turn the large tensor of motion codes into a list of tensors of size "16" to match stage 1 reconstruction
    downsample_factor =  _stage2_runtime_cache["stage1_model"].downsample_factor
    chunk_codes = (chunk_frames // downsample_factor)
    stride = _stage2_runtime_cache["segmenter"].stride # stride value for reconstruction

    # Sliding window chunking to keep stride overlap
    token_chunks = pose_codes.unfold(dimension=1, size=chunk_codes, step=stride // downsample_factor)
    token_chunks = token_chunks.permute(1, 0, 3, 2)
    pose_codes_list = list(token_chunks.unbind(dim=0)) # Permute from [batch, num_chunks, chunk_frames, downsample_factor] to a list of [batch, chunk_frames, downsample_factor] tensors

    # Update the main variables to match the proper iterations of the reconstruction loop
    num_chunks = len(pose_codes_list)
    valid_frames = (num_chunks - 1) * stride + chunk_frames
    full_body_joint_count = 43
    final_reconstructed_6d = torch.zeros((batch_size, valid_frames, full_body_joint_count, 6), device=device)
    pre_frames = chunk_frames - stride
    weight_mask = torch.zeros((batch_size, valid_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=pre_frames, device=device).view(1, -1, 1, 1)

    print("="*50)
    print(f"Num chunks is of size {num_chunks} and the list is of size {len(pose_codes_list)} and pose_codes has a shape of {pose_codes.shape} and an entry in the list has shape {pose_codes_list[0].shape}")
    print(f"{pose_codes_list}")
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
            warm_start = gr.Checkbox(value=True, label="Warm-Start (Seed with ground-truth context)")
            
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
            inputs=[target_npz_file, gr.State(device), top_k, top_p, temperature, warm_start],
            outputs=[output_video_component, pipeline_status]
        )