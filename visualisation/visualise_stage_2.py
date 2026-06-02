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
        print("-------HERE----------")
        print(stage2_config_path, stage2_checkpoint_path, segmenter_config_path)
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
        print("-------HEREe----------")
        
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

        print("-------HEREeeee----------")
        
        stage2_model.load_state_dict(torch.load(stage2_checkpoint_path, map_location=device))
        print("-------HEEEEERE----------")
        stage2_model.eval()
        
        # Additional cache info
        _stage2_runtime_cache["stage1_model"] = stage1_model
        _stage2_runtime_cache["stage2_model"] = stage2_model
        _stage2_runtime_cache["max_seq_len"] = args.model_params.max_seq_len
        _stage2_runtime_cache["code_dim"] = m_s1["code_dim"]
        _stage2_runtime_cache["downsample_factor"] = m_s1.get("downsample_factor", 4)

        print("-------HEERERERE----------")
        
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
            print("-------HEREYUYYY----------")
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
    if original_loader.meta_data.partner_full_id:
        partner_filename = f"{original_loader.meta_data.partner_full_id}.npz"
        partner_path = Path(target_npz_path).parent / partner_filename
        if partner_path.exists():
            partner_loader = utils.SeamlessSMPLHLoader(str(partner_path), device=device, with_meta_data=True, load_vertices=False)

    # use the segmenter for processed data
    segment_data = segmenter._get_loader_segmentation_data(original_loader, partner_loader)

    # reformat arrays
    agent_audio_features = torch.from_numpy(segment_data.aligned_audio).unsqueeze(0).to(device)
    bert_text_features = torch.from_numpy(np.array([word_info["embedding"] for word_info in segment_data.aligned_words])).unsqueeze(0).to(device)
    conversation_states = torch.from_numpy(segment_data.aligned_conversation_states).unsqueeze(0).to(device)
    personality_traits = torch.from_numpy(segment_data.personality_vector).unsqueeze(0).to(device)
    
    # Get partner audio
    if partner_loader is not None:
        partner_segment_data = segmenter._get_loader_segmentation_data(partner_loader, original_loader)
        partner_audio_features = torch.from_numpy(partner_segment_data.aligned_audio).unsqueeze(0).to(device)
    else:
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
    
    generated_tokens = torch.zeros((1, total_token_frames, 4), dtype=torch.long, device=device)
    
    # Generate tokens autoregressively over the continuous timeline (Stage 2)
    for t in range(1, total_token_frames):
        if t % 20 == 0:
            progress(0.1 + 0.5 * (t / total_token_frames), desc=f"Generating gesture tokens: Frame {t}/{total_token_frames}")
            
        start_index = max(0, t - max_sequence_length)
        
        # Window slicing
        past_motion_window = generated_tokens[:, start_index:t, :]
        conversation_states_window = conversation_states_downsampled[:, start_index:t]
        emotion_categories_window = emotion_categories_downsampled[:, start_index:t]
        agent_audio_window = agent_audio_downsampled[:, start_index:t]
        partner_audio_window = partner_audio_downsampled[:, start_index:t]
        bert_text_window = bert_text_downsampled[:, start_index:t]
        
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

    progress(0.6, desc="Decoding token tracks via Stage 1 windowed chunks...")
    
    # Downsampled sliding-window token decoding and continuous weight blending (Stage 1 reconstruction using stage 2 motion codes)
    native_chunk_frames = segmenter.pose_length
    native_stride = segmenter.stride
    
    token_chunk_frames = native_chunk_frames // downsample_factor
    token_stride = native_stride // downsample_factor
    token_pre_frames = token_chunk_frames - token_stride
    
    num_chunks = math.floor((total_token_frames - token_chunk_frames) / token_stride) + 1
    valid_token_frames = (num_chunks - 1) * token_stride + token_chunk_frames
    native_valid_frames = (num_chunks - 1) * native_stride + native_chunk_frames
    
    full_body_joint_count = 43
    batch_size = 1
    
    final_reconstructed_6d = torch.zeros((batch_size, valid_token_frames, full_body_joint_count, 6), device=device)
    weight_mask = torch.zeros((batch_size, valid_token_frames, 1, 1), device=device)
    fade_in = torch.linspace(0.0, 1.0, steps=token_pre_frames, device=device).view(1, -1, 1, 1)
    fade_out = torch.linspace(1.0, 0.0, steps=token_pre_frames, device=device).view(1, -1, 1, 1)
    
    latent_last = None
    
    for i in range(num_chunks):
        start_token_frame = i * token_stride
        end_token_frame = start_token_frame + token_chunk_frames
        chunk_tokens = generated_tokens[:, start_token_frame:end_token_frame, :]
        
        # Decode chunk tokens into 6D rotations -> Shape: [1, 16, 258]
        decoded_pose_6d = stage1_model.decode(chunk_tokens)
        decoded_pose_6d = decoded_pose_6d.reshape(batch_size, token_chunk_frames, full_body_joint_count, 6)
        
        if i > 0 and latent_last is not None:
            decoded_pose_6d[:, :token_pre_frames, :, :] = latent_last[:, -token_pre_frames:, :, :]
            
        reconstructed_matrix_chunk = rc.rotation_6d_to_matrix(decoded_pose_6d)
        decoded_pose_6d = rc.matrix_to_rotation_6d(reconstructed_matrix_chunk)
        latent_last = decoded_pose_6d.clone()
        
        chunk_weight = torch.ones((batch_size, token_chunk_frames, full_body_joint_count, 6), device=device)
        if i > 0:
            chunk_weight[:, :token_pre_frames, :, :] *= fade_in
        if i < num_chunks - 1:
            chunk_weight[:, -token_pre_frames:, :, :] *= fade_out
            
        final_reconstructed_6d[:, start_token_frame:end_token_frame] += decoded_pose_6d * chunk_weight
        weight_mask[:, start_token_frame:end_token_frame] += chunk_weight[:, :, :1, :1]
    
    final_reconstructed_6d = final_reconstructed_6d / (weight_mask + 1e-8)
    reconstructed_6d = final_reconstructed_6d # [1, num_frames, 43, 6]
    reconstructed_matrix = rc.rotation_6d_to_matrix(reconstructed_6d)
    final_reconstructed_pose = rc.matrix_to_axis_angle(reconstructed_matrix).reshape(batch_size * native_valid_frames, -1) # [frames, channels]
    
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
            stage2_config = create_file_browser("Stage 2 Config Path (.json)", extensions=[".json"], file_count="single")
            stage2_checkpoint = create_file_browser("Stage 2 Checkpoint Path (.bin)", extensions=[".bin"], file_count="single")
            segmenter_config = create_file_browser("Segmenter Config Path (.json)", extensions=[".json"], file_count="single")
            
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
            target_npz_file = create_file_browser("Target .NPZ Sequence File Path", extensions=[".npz"], file_count="single")
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