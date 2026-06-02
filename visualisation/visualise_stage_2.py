import gradio as gr
import torch
import torch.nn.functional as F
import json
import math
import os
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from stage_2_models.turn_taking_transformer import TurnTakingTransformer
from training.stage2_trainer_seamless import load_stage_1_checkpoint
from utils import rotation_conversions as rc
import data_processing.utils as utils
from data_processing.seamless_segmenter import SeamlessSegmenter
from data_processing.seamless_segmenter import relationship_detail_map
from visualisation.file_browser import create_file_browser

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
        args.model_params = SimpleNamespace(**config_dict["model_params"])
        args.stage_1_pretrain = SimpleNamespace(**config_dict["stage_1_pretrain"])
        
        # Load stage 1
        m_s1, stage1_model = load_stage_1_checkpoint(
            args.stage_1_pretrain.checkpoint_path,
            args.stage_1_pretrain.config_path,
            device
        )
        
        # Load stage 2
        stage2_model = TurnTakingTransformer(
            vocab_size=m_s1["code_dim"],
            d_model=args.model_params.d_model,
            nhead=args.model_params.nhead,
            num_layers=args.model_params.num_layers,
            dim_feedforward=args.model_params.dim_feedforward,
            dropout=args.model_params.dropout,
            max_seq_len=args.model_params.max_seq_len,
            activation=args.model_params.activation
        ).to(device)
        
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

@torch.no_grad()
def run_autoregressive_generation_pipeline(target_npz_path, device_str, top_k, top_p, temperature, progress=gr.Progress()):
    """
    generates discrete gesture codebook codes frame-by-frame,
    then uses Stage 1 to decode them back into smpl-h motion.
    """
    stage2_model = _stage2_runtime_cache["stage2_model"]
    stage1_model = _stage2_runtime_cache["stage1_model"]
    segmenter = _stage2_runtime_cache["segmenter"]
    
    if stage2_model is None or stage1_model is None or segmenter is None:
        return None, "Status: Please initialize the models first using step 1."

    device = torch.device(device_str)
    progress(0.1, desc="Extracting features using SeamlessSegmenter...")
    
    # Initialize asset dataset and loaders
    original_loader = utils.SeamlessSMPLHLoader(target_npz_path, device=device, with_meta_data=True, load_vertices=True)
    partner_loader = None
    if original_loader.meta_data and original_loader.meta_data.partner_full_id:
        # Recursively look for the partner file
        search_root = Path(target_npz_path).parents[2]
        disk_map = {path.stem: str(path.absolute()) for path in search_root.rglob("*.npz")}
        
        partner_path = disk_map.get(original_loader.meta_data.partner_full_id)
        if partner_path:
            partner_loader = utils.SeamlessSMPLHLoader(partner_path, device=device, with_meta_data=True, load_vertices=False)

    # use the segmenter for processed data
    segment_data = segmenter._get_loader_segmentation_data(original_loader, partner_loader)

    # reformat arrays
    agent_audio_features = torch.from_numpy(segment_data.aligned_audio).unsqueeze(0).to(device).float()
    bert_text_features = torch.from_numpy(np.array([word_info["embedding"] for word_info in segment_data.aligned_words])).unsqueeze(0).to(device).float()
    conversation_states = torch.from_numpy(segment_data.aligned_conversation_states).unsqueeze(0).to(device)
    personality_traits = torch.from_numpy(segment_data.personality_vector).unsqueeze(0).to(device).float()
    
    # Get partner audio
    if partner_loader is not None:
        partner_segment_data = segmenter._get_loader_segmentation_data(partner_loader, original_loader)
        partner_audio_features = torch.from_numpy(partner_segment_data.aligned_audio).unsqueeze(0).to(device).float()
    else:
        print("\nWARNING: PARTNER WAS NOT FOUND! USING ZERO DATA FOR PARTNER AUDIO\n")
        partner_audio_features = torch.zeros_like(agent_audio_features)
        
    # Map the relationship status
    relationship_string = str(original_loader.meta_data.relationship_detail).lower().strip() if original_loader.meta_data.relationship_detail else "stranger"
    social_relationship = torch.tensor([relationship_detail_map.get(relationship_string, 0)], dtype=torch.long, device=device)
    
    emotion_scores = original_loader.meta_data.emotion_scores
    if emotion_scores is not None:
        emotion_categories = torch.from_numpy(np.argmax(emotion_scores, axis=1)).long().unsqueeze(0).to(device)
    else:
        emotion_categories = torch.full((1, original_loader.num_frames), 5, dtype=torch.long, device=device)

    # downsample to fit token space
    downsample_factor = _stage2_runtime_cache["downsample_factor"]
    agent_audio_downsampled = agent_audio_features[:, ::downsample_factor]
    partner_audio_downsampled = partner_audio_features[:, ::downsample_factor]
    bert_text_downsampled = bert_text_features[:, ::downsample_factor]
    conversation_states_downsampled = conversation_states[:, ::downsample_factor]
    emotion_categories_downsampled = emotion_categories[:, ::downsample_factor]
    
    total_token_frames = agent_audio_downsampled.shape[1]
    max_sequence_length = _stage2_runtime_cache["max_seq_len"]

    # Extract global ground truth tokens for window extraction reference
    full_pose_sequence = original_loader.get_trainable_pose().to(device)
    batch_size_gt = 1
    frames_gt = full_pose_sequence.shape[0]
    gt_pose_matrix = rc.axis_angle_to_matrix(full_pose_sequence.reshape(batch_size_gt, frames_gt, 43, 3))
    gt_pose_6d = rc.matrix_to_rotation_6d(gt_pose_matrix).reshape(batch_size_gt, frames_gt, 258)
    gt_tokens = stage1_model.encode(gt_pose_6d)  # Shape: [1, total_token_frames, 4]
    
    generated_tokens = torch.zeros((1, total_token_frames, 4), dtype=torch.long, device=device)
    
    # Track evaluation metrics inside independent 64-frame windows
    total_eval_frames = 0
    match_counts = [0, 0, 0, 0]
    strict_hit_count = 0
    
    # Split the long timeline into independent 64-frame chunks matching training constraints
    for chunk_start in range(0, total_token_frames, max_sequence_length):
        chunk_end = min(chunk_start + max_sequence_length, total_token_frames)
        chunk_len = chunk_end - chunk_start
        if chunk_len < 2:
            continue
            
        # Seed only the very first frame of the chunk with ground truth
        generated_tokens[:, chunk_start, :] = gt_tokens[:, chunk_start, :]
        
        # Autoregressively generate the remainder of this specific 64-frame window
        for t in range(chunk_start + 1, chunk_end):
            if t % 20 == 0:
                progress(0.1 + 0.5 * (t / total_token_frames), desc=f"Generating gesture tokens: Frame {t}/{total_token_frames}")
            
            # Keep lookback window locked locally to the start of this chunk
            past_motion_window = generated_tokens[:, chunk_start:t, :]
            conversation_states_window = conversation_states_downsampled[:, chunk_start:t]
            emotion_categories_window = emotion_categories_downsampled[:, chunk_start:t]
            agent_audio_window = agent_audio_downsampled[:, chunk_start:t]
            partner_audio_window = partner_audio_downsampled[:, chunk_start:t]
            bert_text_window = bert_text_downsampled[:, chunk_start:t]
            
            prediction_logits, _ = stage2_model(
                past_motion_codes=past_motion_window,
                vad_integers=conversation_states_window,
                emotion_categories=emotion_categories_window,
                social_relationship=social_relationship,
                personality_traits=personality_traits,
                audio_features=agent_audio_window,
                partner_audio_features=partner_audio_window,
                bert_features=bert_text_window
            )
            
            next_frame_logits = prediction_logits[:, -1, :, :]
            
            for layer_idx in range(4):
                logits_layer = next_frame_logits[0, layer_idx, :]
                sampled_id = sample_top_p_k(logits_layer, top_k=top_k, top_p=top_p, temperature=temperature)
                generated_tokens[0, t, layer_idx] = sampled_id.squeeze(-1)
                
            # Accumulate evaluation statistics frame by frame within this window
            total_eval_frames += 1
            is_strict_match = True
            for layer_idx in range(4):
                if generated_tokens[0, t, layer_idx] == gt_tokens[0, t, layer_idx]:
                    match_counts[layer_idx] += 1
                else:
                    is_strict_match = False
            if is_strict_match:
                strict_hit_count += 1

    # ==================== CLEAN DIAGNOSTICS ENGINE ====================
    print("\n" + "="*50)
    print("      STAGE 2 64-FRAME SLIDING WINDOW EVALUATION")
    print("="*50)
    print(f"Evaluating across independent window segments ({total_eval_frames} total predicted steps):")
    for layer_idx in range(4):
        layer_acc = (match_counts[layer_idx] / total_eval_frames) * 100.0
        print(f"  -> Codebook Depth {layer_idx + 1} Match Accuracy: {layer_acc:.2f}% ({match_counts[layer_idx]}/{total_eval_frames})")
    print(f"  -> Absolute Sequence Frame Hit Rate (Strict Match): {(strict_hit_count / total_eval_frames) * 100.0:.2f}%")
    print("="*50 + "\n")
    # =============================================================================

    progress(0.6, desc="Decoding token tracks via Stage 1 windowed chunks...")
    
    # Downsampled sliding-window token decoding and continuous weight blending
    native_chunk_frames = segmenter.pose_length
    native_stride = segmenter.stride
    native_pre_frames = native_chunk_frames - native_stride
    
    token_chunk_frames = native_chunk_frames // downsample_factor
    token_stride = native_stride // downsample_factor
    
    num_chunks = math.floor((total_token_frames - token_chunk_frames) / token_stride) + 1
    
    native_valid_frames = (num_chunks - 1) * native_stride + native_chunk_frames
    
    full_body_joint_count = 43
    batch_size = 1
    
    final_reconstructed_6d = torch.zeros((batch_size, native_valid_frames, full_body_joint_count, 6), device=device)
    weight_mask = torch.zeros((batch_size, native_valid_frames, 1, 1), device=device)
    fade_in_native = torch.linspace(0.0, 1.0, steps=native_pre_frames, device=device).view(1, -1, 1, 1)
    fade_out_native = torch.linspace(1.0, 0.0, steps=native_pre_frames, device=device).view(1, -1, 1, 1)
    
    latent_last = None
    
    for i in range(num_chunks):
        start_native_frame = i * native_stride
        end_native_frame = start_native_frame + native_chunk_frames
        
        start_token_frame = i * token_stride
        end_token_frame = start_token_frame + token_chunk_frames
        chunk_tokens = generated_tokens[:, start_token_frame:end_token_frame, :]
        
        # Decode chunk tokens into 6D rotations -> Shape: [1, 64, 258]
        decoded_pose_6d = stage1_model.decode(chunk_tokens)
        decoded_pose_6d = decoded_pose_6d.reshape(batch_size, native_chunk_frames, full_body_joint_count, 6)
        
        if i > 0 and latent_last is not None:
            decoded_pose_6d[:, :native_pre_frames, :, :] = latent_last[:, -native_pre_frames:, :, :]
            
        reconstructed_matrix_chunk = rc.rotation_6d_to_matrix(decoded_pose_6d)
        decoded_pose_6d = rc.matrix_to_rotation_6d(reconstructed_matrix_chunk)
        latent_last = decoded_pose_6d.clone()
        
        chunk_weight = torch.ones((batch_size, native_chunk_frames, full_body_joint_count, 6), device=device)
        
        if i > 0:
            chunk_weight[:, :native_pre_frames, :, :] *= fade_in_native
        if i < num_chunks - 1:
            chunk_weight[:, -native_pre_frames:, :, :] *= fade_out_native
            
        final_reconstructed_6d[:, start_native_frame:end_native_frame] += decoded_pose_6d * chunk_weight
        weight_mask[:, start_native_frame:end_native_frame] += chunk_weight[:, :, :1, :1]
    
    final_reconstructed_6d = final_reconstructed_6d / (weight_mask + 1e-8)
    reconstructed_6d = final_reconstructed_6d
    reconstructed_matrix = rc.rotation_6d_to_matrix(reconstructed_6d)
    final_reconstructed_pose = rc.matrix_to_axis_angle(reconstructed_matrix).reshape(batch_size * native_valid_frames, -1)
    
    # Overwrite the loader with predicted results
    reconstructed_loader = utils.SeamlessSMPLHLoader(target_npz_path, device=device, with_meta_data=False, load_vertices=False)
    reconstructed_loader.overwrite_with_reconstructed_pose(final_reconstructed_pose)
    
    progress(0.9, desc="Rendering comparative animation window video...")
    output_video_path = utils.generate_reconstruction_comparison_video(original_loader, reconstructed_loader, progress=progress, use_cache=False)
    
    return output_video_path, "Status: Generation and Rendering Complete!"

def render_stage2_generation_tab(device):
    with gr.TabItem("Stage 2: Audio Gesture Generation"):
        gr.Markdown("### 1. Load System Framework (Stage 2 + Stage 1 Dependency)")
        with gr.Row():
            stage2_config = create_file_browser("Stage 2 Config Path (.json)", extensions=[".json"], file_count="single", default_value="/data/scratch/davwer_projects/mpi-thesis/configs/stage2_model.json")
            stage2_checkpoint = create_file_browser("Stage 2 Checkpoint Path (.bin)", extensions=[".bin"], file_count="single", default_value="/data/scratch/davwer_projects/mpi-thesis/checkpoints_stage_2/stage_2_model/20260601_1840/stage2_final_epoch_199.bin")
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
            target_npz_file = create_file_browser("Target .NPZ Sequence File Path", extensions=[".npz"], file_count="single", default_value="data/seamless/downloads/stage_2_data_imitator/naturalistic/train/0002/0001/V00_S0124_I00000384_P0183.npz")
            generate_gesture_btn = gr.Button("Generate Gestures from Audio Context", variant="primary")
            
        output_video_component = gr.Video(label="Front View Evaluation: Ground Truth (Left) | Generated from Audio (Right)")

        load_pipeline_btn.click(
            fn=load_stage2_and_stage1_pipeline,
            inputs=[stage2_config, stage2_checkpoint, segmenter_config, gr.State(device)],
            outputs=pipeline_status
        )
        
        generate_gesture_btn.click(
            fn=run_autoregressive_generation_pipeline,
            inputs=[target_npz_file, gr.State(device), top_k, top_p, temperature],
            outputs=[output_video_component, pipeline_status]
        )