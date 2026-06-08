import lmdb
import pickle
import torch
import librosa
import numpy as np
import math
from pathlib import Path
from loguru import logger
from .seamless_dataset import SeamlessDataset
from .utils import SeamlessSMPLHLoader
from transformers import AutoTokenizer, BertModel, Wav2Vec2Processor, HubertModel
import torch.nn.functional as F
import json
import argparse
import torch.multiprocessing as mp
import os
from datetime import datetime
from dataclasses import dataclass
from numpy.typing import NDArray
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cache_dir = PROJECT_ROOT / "cache" / "segmenter"

relationship_detail_map = {
    "stranger": 0,
    "coworkers": 1,
    "friends": 2,
    "dating/spouse/romantic_partner": 3,
    "siblings": 4,
    "parent_child": 5,
    "roommates": 6,
    "classmates": 7,
    "neighbors": 8,
    "familiar-generic": 9,
    "family-generic": 10
}

@dataclass
class LoaderSegmentationData:
    aligned_pose : any = None
    aligned_raw_audio : any = None
    aligned_wav2vec_audio : any = None
    aligned_hubert_audio : any = None
    aligned_words : any = None
    aligned_emotion_list : any = None
    aligned_semantic_values : any = None
    aligned_off_screen_hand_values : any = None
    num_subdivisions : any = None
    aligned_conversation_states: any = None
    personality_vector: any = None


class SeamlessSegmenter:
    def __init__(self, dataset : SeamlessDataset, pose_fps=30, audio_sr=16000, audio_hz=50, pose_length=64, stride=10, seed=42, max_gb=10,
                 clean_first_seconds=0, clean_final_seconds=0, segment_mode='both', 
                 filter_out_of_screen_hands=False, keep_only_high_intensity_blocks=False, vad_padding_in_seconds=0, worker_id=-1,
                 project_name="unknown", keep_listener_backchannel=False, include_stage_2_partner_data=False, 
                 relationship_detail_mandatory=False, personality_detail_mandatory=False, single_worker_mode=False):

        self.dataset = dataset
        dataset_name = getattr(dataset, 'name', 'unknown_dataset')
        if worker_id == -1:
            self.output_path = cache_dir / project_name / f"{dataset_name}"
        else:
            self.output_path = cache_dir / project_name / f"{dataset_name}" / f"part_{worker_id}"

        self.pose_fps = pose_fps
        self.audio_hz = audio_hz
        self.audio_sr = audio_sr
        self.pose_length = pose_length
        self.stride = stride
        self.max_gb = max_gb
        self.worker_id = worker_id

        self.clean_first = clean_first_seconds
        self.clean_final = clean_final_seconds
        self.segment_mode = segment_mode # Can be, one of the following [both, listening, speaking] (segments the frames to only get that data)
        self.filter_out_of_screen_hands = filter_out_of_screen_hands # used to filter out any frames where the hand is out of the screen (noise)
        self.keep_only_high_intensity_blocks = keep_only_high_intensity_blocks # This is only relevant for 'listener' segment mode, it will keep only high intensity VAD frames (VAD with movement)
        self.vad_padding_in_seconds = vad_padding_in_seconds # this is to not only grab the VAD section for the speaker mode but also pad it with the build up to the VAD
        self.vad_padding_in_frames = int(self.vad_padding_in_seconds * self.pose_fps)
        self.keep_listener_backchannel = keep_listener_backchannel
        self.include_stage_2_partner_data = include_stage_2_partner_data
        self.relationship_detail_mandatory = relationship_detail_mandatory
        self.personality_detail_mandatory = personality_detail_mandatory
        
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").eval()

        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").eval()

    def build_cache(self):
        """Processes all interactions in a given seamless interaction dataset and writes the segments to LMDB."""
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

        # create/reserve the lmdb cache
        env = lmdb.open(str(self.output_path), map_size=int(1024**3 * self.max_gb))
        
        sample_idx = 0
        # Loop over each sample
        total_samples = len(self.dataset.samples)
        items = list(self.dataset.samples.items())
        for interaction_idx, (_, _) in enumerate(items):
            # Get a seamless loader, of the data sample for easy access to all needed information
            partner_loader = None
            if self.include_stage_2_partner_data:
                loader, partner_loader = self.dataset.get_interaction_pair(self.dataset.file_ids[interaction_idx], {'load_vertices': False, 'with_meta_data': True})
                if partner_loader is None:
                    logger.warning(f"No partner found for {interaction_idx}")
                    continue
                if loader.meta_data.relationship_detail is None and self.relationship_detail_mandatory:
                    logger.warning(f"No relationship_detail found for {interaction_idx}")
                    continue
                detail_str = str(loader.meta_data.relationship_detail).lower().strip()
                relationship_id = relationship_detail_map.get(detail_str, 0)
            else:
                loader = self.dataset.get_interaction(self.dataset.file_ids[interaction_idx], {'load_vertices': False, 'with_meta_data': True})
            
            loader_data = self._get_loader_segmentation_data(loader, partner_loader)
            # Get the correct amount of subdivisions for this interaction
            loop_subdivisions = loader_data.num_subdivisions
            if self.include_stage_2_partner_data:
                partner_loader_data = self._get_loader_segmentation_data(partner_loader, loader)
                loop_subdivisions = min(loader_data.num_subdivisions, partner_loader_data.num_subdivisions)
            if self.personality_detail_mandatory and np.all(loader_data.personality_vector == 0.0):
                logger.warning(f"No personality info found for {interaction_idx}")
                continue

            # Write to the database
            with env.begin(write=True) as transaction:
                for i in range(loop_subdivisions):
                    # motion segment
                    pose_start = i * self.stride
                    pose_end = pose_start + self.pose_length

                    # For audio
                    start_time = pose_start / self.pose_fps
                    audio_start = int(start_time * self.audio_hz) 
                    target_audio_len = math.floor(self.pose_length * (self.audio_hz / self.pose_fps))
                    audio_end = audio_start + target_audio_len # We do it this way to prevent differnt length audio samples
                    #
                    # For Raw Audio
                    raw_audio_start = int(start_time * self.audio_sr) 
                    target_raw_audio_len = math.floor(self.pose_length * (self.audio_sr / self.pose_fps))
                    raw_audio_end = raw_audio_start + target_raw_audio_len

                    # Perhaps skip this fragment if it has any offscreen hand values and we set to filter those out
                    if self.filter_out_of_screen_hands:
                        sample_offscreen_hand_values = loader_data.aligned_off_screen_hand_values[pose_start:pose_end]
                        if np.any(sample_offscreen_hand_values > 0):
                            continue

                    # semantic values segment
                    sample_semantic_values = loader_data.aligned_semantic_values[pose_start:pose_end]
                    
                    # Filter (if needed) to only listener or speaker frames. This works using the semantic values where 0.0 means there is no VAD
                    if self.segment_mode != 'both':
                        has_speech_core = np.any(sample_semantic_values >= 0.1) 
                        has_padding = np.any(sample_semantic_values == 0.05)
                        if self.segment_mode == 'speaking' and not (has_speech_core or has_padding): # All 0s, so skip
                            continue
                        elif self.segment_mode == 'listening' and has_speech_core: # no 0s or 0.05 so speaking actively
                            # Check if speech is a short backchannel (smaller than 1 second of sound) with low intensity (no 0.2)
                            is_backchannel = (np.sum(sample_semantic_values == 0.1) < self.pose_fps) and not np.any(sample_semantic_values == 0.2)
                            if not is_backchannel or not self.keep_listener_backchannel:
                                continue

                    sample_pose = loader_data.aligned_pose[pose_start:pose_end]
                    
                    # audio segment
                    sample_raw_audio = loader_data.aligned_raw_audio[raw_audio_start:raw_audio_end]
                    sample_wav2vec_audio = loader_data.aligned_wav2vec_audio[audio_start:audio_end]
                    sample_hubert_audio = loader_data.aligned_hubert_audio[audio_start:audio_end]


                    # word segment
                    sample_words = loader_data.aligned_words[pose_start:pose_end]
                    sample_bert = np.array([f["embedding"] for f in sample_words])
                    sample_word_text = np.array([f["word"] for f in sample_words])
                    sample_sentence = np.array([f["sentence"] for f in sample_words])

                    # emotion segment
                    sample_emotions = loader_data.aligned_emotion_list[pose_start:pose_end]

                    # storage setup
                    entry = {
                        "pose": sample_pose.numpy(), # pose segment of 129 dimension
                        "raw_audio": sample_raw_audio, # audio segment
                        "wav2vec_audio": sample_wav2vec_audio, # audio segment
                        "hubert_audio": sample_hubert_audio, # audio segment
                        "bert": sample_bert, # word segments
                        "emotions": sample_emotions, # seamless interaction categories 0-7 mentioned in the helper function
                        "semantic_scores": sample_semantic_values, # semantic values for each frame being 0.0, 0.1 or 1.0
                        "sample_word": sample_word_text, # the actual words in text for each frame of the sample
                        "sample_sentence": sample_sentence, # the sentence for each frame, where the word came from
                        "segment_index": i, # the index of this segment for this specific file
                        "seamless_id": loader.meta_data.full_id, # video/file ID
                        "file_path": loader.path, # the path this file came from
                    }
                    if self.include_stage_2_partner_data:
                        entry["partner_seamless_id"] = loader.meta_data.partner_full_id
                        entry["partner_raw_audio"] = partner_loader_data.aligned_raw_audio[raw_audio_start:raw_audio_end]
                        entry["partner_wav2vec_audio"] = partner_loader_data.aligned_wav2vec_audio[audio_start:audio_end]
                        entry["partner_hubert_audio"] = partner_loader_data.aligned_hubert_audio[audio_start:audio_end]
                        entry["conversation_states"] = loader_data.aligned_conversation_states[pose_start:pose_end]
                        entry["partner_conversation_states"] = partner_loader_data.aligned_conversation_states[pose_start:pose_end]
                        entry["social_relationship"] = relationship_id
                        entry["personality_traits"] = loader_data.personality_vector
                        entry["partner_emotions"] = partner_loader_data.aligned_emotion_list[pose_start:pose_end]
                        entry["partner_personality_traits"] = partner_loader_data.personality_vector
                        entry["partner_pose"] = partner_loader_data.aligned_pose[pose_start:pose_end].numpy()
                        partner_sample_words = partner_loader_data.aligned_words[pose_start:pose_end]
                        partner_sample_bert = np.array([f["embedding"] for f in partner_sample_words])
                        entry["partner_bert"] = partner_sample_bert
                    
                    key = f"{sample_idx:08d}".encode("ascii")
                    serialized_entry = pickle.dumps(entry)
                    transaction.put(key, serialized_entry)
                    sample_idx += 1

            # Logging
            ten_percent_step = max(1, total_samples // 10)
            if (interaction_idx + 1) % ten_percent_step == 0 or (interaction_idx + 1) == total_samples:
                percent = int(((interaction_idx + 1) / total_samples) * 100)
                logger.info(f"Worker {self.worker_id} | Progress: {percent}% | {interaction_idx + 1}/{total_samples} files")

        env.close()
        logger.info(f"Finished worker {self.worker_id}. Total segments cached: {sample_idx}")

    def _get_loader_segmentation_data(self, loader, partner_loader=None):
        device = next(self.bert_model.parameters()).device
        data = LoaderSegmentationData()
        full_pose = loader.get_trainable_pose() # (frames, 129)
        num_frames_target = full_pose.shape[0]
        # load and resample Audio
        audio_data, _ = librosa.load(loader.meta_data.audio_file, sr=self.audio_sr)

        # Compute Full-Length HuBERT Embeddings
        wav2vec_values = self.wav2vec2_processor(audio_data, return_tensors="pt", sampling_rate=self.audio_sr).input_values.to(device)
        self.hubert_model.to(device)
        with torch.no_grad():
            # Extract raw hidden state sequence [1, 50Hz_Frames, 1024]
            audio_embeddings = self.hubert_model(wav2vec_values).last_hidden_state.squeeze(0).cpu().numpy()
        wav2vec_data = wav2vec_values.squeeze(0).cpu().numpy()

        # get the lookup table for the bert word vectors
        bert_lookup_table = self._get_bert_transcript(loader.meta_data.transcript)
        # get the word vectors list len(frames) containing for each index {'word': word, 'sentence': sentence, 'embedding': vector}
        word_list = self._get_word_list(loader, bert_lookup_table)
        # get the emotion category for each frame
        emotion_category_frames = self._get_frame_emotions(loader)
        # get the semantic value for each frame
        semantic_values_list = self._get_frame_semantics(loader)
        # Conversation status
        raw_conversation_states = self._calculate_conversation_states(loader, partner_loader)
        
        # align the audio and pose
        data.aligned_pose, data.aligned_raw_audio, data.aligned_wav2vec_audio, data.aligned_hubert_audio, data.aligned_words, data.aligned_emotion_list, data.aligned_semantic_values, data.aligned_off_screen_hand_values, data.aligned_conversation_states = self._align_data(
            full_pose, audio_data, wav2vec_data, audio_embeddings, word_list, emotion_category_frames, semantic_values_list, loader.meta_data.hand_offscreen, raw_conversation_states
        )

        # see how many 64-frame chunks fit
        data.num_subdivisions = math.floor((len(data.aligned_pose) - self.pose_length) / self.stride) + 1

        # personality vector
        data.personality_vector = np.array([
            loader.meta_data.extraversion_raw or 0.0,
            loader.meta_data.agreeableness_raw or 0.0,
            loader.meta_data.conscientiousness_raw or 0.0,
            loader.meta_data.neuroticism_raw or 0.0,
            loader.meta_data.openness_raw or 0.0
        ], dtype=np.float32)
        return data
    
    def _align_data(self, full_pose, raw_audio, wav2vec_audio_vector, hubert_audio_vectors, word_list, emotion_list, semantic_values_list, off_screen_hand_values, raw_conversation_states):
        """
        Aligns the motion .npz and audio .wav by finding the shortest common duration 
        """
        round_seconds_pose = full_pose.shape[0] / self.pose_fps
        round_seconds_audio = hubert_audio_vectors.shape[0] / self.audio_hz

        round_seconds_common = min(round_seconds_pose, round_seconds_audio)
        
        # (skip first/last x amount of seconds)
        clipose_start_time = self.clean_first
        clipose_end_time = round_seconds_common - self.clean_final
        
        # pose indices
        pose_start = int(clipose_start_time * self.pose_fps)
        pose_end = int(clipose_end_time * self.pose_fps)

        audio_start = int(clipose_start_time * self.audio_hz)
        audio_end = int(clipose_end_time * self.audio_hz)

        raw_audio_start = int(clipose_start_time * self.audio_sr)
        raw_audio_end = int(clipose_end_time * self.audio_sr)

        # slice the arrays to the aligned and cleaned window
        aligned_pose = full_pose[pose_start:pose_end]
        aligned_word_list = word_list[pose_start:pose_end]
        aligned_emotion_list = emotion_list[pose_start:pose_end]
        aligned_raw_audio = raw_audio[raw_audio_start:raw_audio_end]
        aligned_wav2vec_audio = wav2vec_audio_vector[audio_start:audio_end]
        aligned_hubert_audio = hubert_audio_vectors[audio_start:audio_end]
        aligned_semantic_values = semantic_values_list[pose_start:pose_end]
        aligned_off_screen_hand_values = off_screen_hand_values[pose_start:pose_end]
        aligned_conversation_states = raw_conversation_states[pose_start:pose_end]
        
        return aligned_pose, aligned_raw_audio, aligned_wav2vec_audio, aligned_hubert_audio, aligned_word_list, aligned_emotion_list, aligned_semantic_values, aligned_off_screen_hand_values, aligned_conversation_states
    
    def _get_word_list(self, loader : SeamlessSMPLHLoader, bert_lookup_table : dict):
        """
        Maps words from the transcript json to each motion frame.
        Returns a list of len(num_frames) where each index contains a dictionairy {word;sentence} with the word and its corresponding sentence 
        """
        num_frames = loader.num_frames
        transcript = loader.meta_data.transcript # json object
        
        # initialize list with a EMPTY token
        zero_vector = np.zeros(768)
        frame_words = [{"word": "EMPTY", "sentence": "EMPTY", "embedding": zero_vector} for _ in range(num_frames)]
        
        # flatten the nested transcript into a simple list of words
        segments_list = transcript.get("metadata:transcript", [])
        for segment in segments_list:
            sentence_text = segment["transcript"]
            # map each word to the corresponding frame start/end time
            for word_info in segment["words"]:
                if word_info.get('start') is None or word_info.get('end') is None:
                    logger.warning(f"Skipping word {word_info.get('word')} due to missing timestamps")
                    continue
                start_frame = int(word_info['start'] * self.pose_fps)
                end_frame = int(word_info['end'] * self.pose_fps)
                
                # clamp to ensure we stay within the actual motion length
                actual_start = max(0, start_frame)
                actual_end = min(num_frames, end_frame)
                
                for f in range(actual_start, actual_end):
                    frame_words[f] = {"word": word_info['word'], "sentence": sentence_text, "embedding": bert_lookup_table.get(sentence_text, {}).get(word_info['word'], zero_vector)}
                
        return frame_words

    def _get_bert_transcript(self, transcript):
        """
        Turns the transcript into bert vectors, it handles each sentence, and converts it to a vector
        For each word it will have a vector
        Returns a dictionairy being {Sentence : {word1: vector, word2: vector}, Sentence : {word1: vector, word2: vector}}
        """
        return_dict = {}
        zero_vector = np.zeros(768)
        segments_list = transcript.get("metadata:transcript", [])
        for segment in segments_list:
            sentence_text = segment["transcript"]
            # encode sentence with BERT and get character offsets
            inputs = self.tokenizer(sentence_text, return_offsets_mapping=True, return_tensors="pt")
            device = next(self.bert_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()} # Move to device
            with torch.no_grad():
                outputs = self.bert_model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
            token_vecs = outputs.last_hidden_state.squeeze(0)
            offsets = inputs['offset_mapping'].squeeze(0).tolist()
            segment_dict = {}
            for word_info in segment["words"]:
                target_word = word_info['word']
                
                # get the position of the word in the sentence
                start_char = sentence_text.find(target_word)
                end_char = start_char + len(target_word)
                
                # get all tokens within the words range
                relevant_token_vecs = []
                for i, (t_start, t_end) in enumerate(offsets):
                    # This also skips (0,0) tokens which are special tokens
                    if t_start >= start_char and t_end <= end_char and t_start != t_end:
                        relevant_token_vecs.append(token_vecs[i])
                
                # Average the vectors to get a general word vector
                if relevant_token_vecs:
                    segment_dict[target_word] = torch.stack(relevant_token_vecs).mean(dim=0).detach().cpu().numpy()
                else:
                    # No tokens found, is EMPTY or just no word, meaning we use a 0's vector
                    segment_dict[target_word] = zero_vector
            return_dict[sentence_text] = segment_dict
        return return_dict

    def _get_frame_emotions(self, loader : SeamlessSMPLHLoader):
        """
        return a list of size num_frames with an integer for each frame representing an emotion
        Categories are in order being, Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
        if no emotion_scores are present, we default to using 5 being Neutral
        """
        emotion_scores = loader.meta_data.emotion_scores
        return np.argmax(emotion_scores, axis=1).astype(np.int32) if emotion_scores is not None else np.full(loader.num_frames, 5, dtype=np.int32)

    def _calculate_conversation_states(self, loader, partner_loader, pose_fps=30):
        """
        Generates a timeline that shows when the target was speaking for stage 2 training
        
        States:
        0: Target Quiet (listening)
        1: Target speaking, partner silent
        2: Both speaking simultaneously
        """
        # Initialize full-duration frame tracks for both participants
        num_frames_target = loader.num_frames
        conversation_states = np.zeros(num_frames_target, dtype=np.int32)
        target_speech_track = np.zeros(num_frames_target, dtype=np.int32)
        
        # Fill target track from raw intervals
        target_vad_list = loader.meta_data.transcript.get('metadata:vad', [])
        for d in target_vad_list:
            s_frame = int(d['start'] * pose_fps)
            e_frame = int(d['end'] * pose_fps)
            target_speech_track[max(0, s_frame):min(num_frames_target, e_frame)] = 1

        # Fill partner track
        if partner_loader is not None:
            num_frames_partner = partner_loader.num_frames
            partner_speech_track = np.zeros(num_frames_partner, dtype=np.int32)
            
            partner_vad_list = partner_loader.meta_data.transcript.get('metadata:vad', [])
            for d in partner_vad_list:
                s_frame = int(d['start'] * pose_fps)
                e_frame = int(d['end'] * pose_fps)
                partner_speech_track[max(0, s_frame):min(num_frames_partner, e_frame)] = 1
        else:
            # If no partner, they are silent across the whole file duration
            partner_speech_track = np.zeros(num_frames_target, dtype=np.int32)
        
        # If partner track is shorter, pad it with zeros, if longer, truncate it.
        if len(partner_speech_track) != num_frames_target:
            aligned_partner = np.zeros(num_frames_target, dtype=np.int32)
            copy_len = min(len(partner_speech_track), num_frames_target)
            aligned_partner[:copy_len] = partner_speech_track[:copy_len]
            partner_speech_track = aligned_partner

        # classify each frame index
        conversation_states[(target_speech_track == 1) & (partner_speech_track == 0)] = 1
        # Both target and partner chunk are 1
        conversation_states[(target_speech_track == 1) & (partner_speech_track == 1)] = 2

        return conversation_states

    def _get_frame_semantics(self, loader : SeamlessSMPLHLoader):
        """
        returns semantic values for each frame
        values can be:
        0.0 for no speaking frames
        0.05 padded listening frames, about to turn into speaking frames
        0.1 for speaking frames
        0.2: high-intensity speaking frames (top 25% movement of this person during the interaction)
        1.0 for any motion of interest frames
        """
        vad_info = loader.meta_data.transcript.get('metadata:vad', None)
        # initialise for 0.0 silence
        sem_scores = np.zeros(loader.num_frames, dtype=np.float32)
        if vad_info != None:
            movement = loader.meta_data.movement_intensity
            threshold = loader.meta_data.movement_intensity_peak_threshold
            for vad_segment in vad_info:
                start_frame = int(vad_segment['start'] * self.pose_fps)
                end_frame = int(vad_segment['end'] * self.pose_fps)
                
                if self.keep_only_high_intensity_blocks and movement is not None:
                    index_start = max(0, start_frame)
                    index_end = min(len(movement), end_frame)
                    
                    # Check for a local peak within this entire VAD block
                    block_max = np.max(movement[index_start:index_end]) if index_start < index_end else 0
                    if block_max <= threshold:
                        continue

                # Mark the entire padded vad sequence as 0.05 (padded)
                if self.vad_padding_in_frames > 0.0:
                    padded_start = max(0, start_frame - self.vad_padding_in_frames)
                    padded_end = min(loader.num_frames, end_frame + self.vad_padding_in_frames)
                    sem_scores[padded_start:padded_end] = 0.05
                # mark the vad sequence as speech (0.1) (overwrites any that were 0.05)
                sem_scores[max(0, start_frame):min(loader.num_frames, end_frame)] = 0.1
        
        # improve 0.1 to 0.2 if movement is in the top 25% (more movement during speech)
        if movement is not None:
            # We only upgrade frames that are already part of a VAD segment
            peak_mask = (loader.meta_data.movement_intensity > threshold) & (sem_scores == 0.1)
            sem_scores[peak_mask] = 0.2

        # Check for any annotations (moments of interest (MOI))
        annotation_keys = [key for key in loader.meta_data.transcript.keys() if "annotations:" in key]
        for key in annotation_keys:
            data = loader.meta_data.transcript[key]
            moment_of_interests = data if isinstance(data, list) else [data] # if the annotations are multiple it will be a list of annotations
            for moi in moment_of_interests:
                if isinstance(moi, dict) and 'start_ts' in moi:
                    start_frame = int(moi['start_ts'] * self.pose_fps)
                    end_frame = int(moi['end_ts'] * self.pose_fps)
                    # If any annotation exists we set it as 1.0 since it has high semantic
                    sem_scores[max(0, start_frame):min(loader.num_frames, end_frame)] = 1.0
        return sem_scores

def run_worker(worker_id, file_ids, config_data, dataset_path, gpu_id):
    """Function that runs the segmenter on a worker."""
    # Set logging for this specific worker
    logger.info(f"Worker {worker_id} starting on GPU {gpu_id} with {len(file_ids)} files.")
    device = torch.device(f"cuda:{gpu_id}")
    
    # Re-initialize the dataset and filter it to only this workers given file_ids
    dataset = SeamlessDataset(PROJECT_ROOT / dataset_path, file_ids=file_ids)

    segmenter = SeamlessSegmenter(dataset, worker_id=worker_id, **config_data)

    segmenter.bert_model.to(device) # Make sure the bert_model is on the right device
    
    # run the segmentation
    segmenter.build_cache()
    logger.info(f"Worker {worker_id} finished.")

    # short check to see if it all worked fine
    env = lmdb.open(str(segmenter.output_path), readonly=True)
    with env.begin() as transaction:
        cursor = transaction.cursor()
        if cursor.first():
            _, raw = cursor.item()
            entry = pickle.loads(raw)
            logger.info(f"{dataset.name} Pose shape: {entry['pose'].shape}") # should be (64, 129)
            logger.info(f"{dataset.name} Raw Audio shape: {entry['raw_audio'].shape}")
            logger.info(f"{dataset.name} wav2vec Audio shape: {entry['wav2vec_audio'].shape}")
            logger.info(f"{dataset.name} hubert Audio shape: {entry['hubert_audio'].shape}")
            logger.info(f"{dataset.name} BERT shape: {entry['bert'].shape}") # should be (64, 768)
            logger.info(f"{dataset.name} Emotions shape: {entry['emotions'].shape}")
            logger.info(f"{dataset.name} Semantics shape: {entry['semantic_scores'].shape}")
        else:
            logger.error(f"Worker {worker_id} | Verification Failed: Database is empty/not found!")

def merge_parts(merged_path, dataset_name, config_data):
    
    # Prevent overwriting an existing merged dataset
    if merged_path.exists() and any(merged_path.glob("*.lmdb")):
        logger.warning(f"Master path {merged_path} already contains an LMDB. Skipping merge to prevent corruption.")
        return
    merged_path.mkdir(parents=True, exist_ok=True)
    
    # Create merged LMDB
    map_size = int(config_data.get('max_gb', 10) * 1024**3)
    merged_env = lmdb.open(str(merged_path), map_size=map_size)
    global_idx = 0

    # Find all part folders experimenting/part_0, experimenting/part_1
    part_paths = sorted(list(merged_path.glob("part_*")))
    
    if not part_paths:
        logger.error(f"No parts found for {dataset_name} in {merged_path.parent}")
        return

    logger.info(f"Merging {len(part_paths)} parts into cache folder...")

    with merged_env.begin(write=True) as merged_txn:
        temp_idx = 0
        for part_path in part_paths:
            part_env = lmdb.open(str(part_path), readonly=True)
            stats = part_env.stat()
            logger.info(f"This subset ({part_path.name}) has: {stats['entries']} entries")
            with part_env.begin() as part_txn:
                cursor = part_txn.cursor()
                for _, value in cursor:
                    # New global key 00000000, 00000001
                    key = f"{global_idx:08d}".encode()
                    merged_txn.put(key, value)
                    temp_idx +=1
                    global_idx += 1
            part_env.close()
            logger.info(f"Successfully merged {temp_idx} total segments from {part_path.name}")
            temp_idx = 0

    merged_env.close()
    for part_path in part_paths:
        logger.info(f"Removing processed partition directory: {part_path.name}")
        shutil.rmtree(part_path)
    logger.info(f"Merge Complete. Final merged dataset at: {merged_path} (Total: {global_idx} segments)")

def main():
    parser = argparse.ArgumentParser(description="Seamless Segmenter Parser")
    parser.add_argument("--dataset_path", type=str, default="data/seamless/downloads/experimenting", help="Path to the dataset folder")
    parser.add_argument("--config", type=str, default="configs/seamless_segmenter.json", help="Path to the segmenter config.json")
    
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--gpu_id", type=int, default=0, help="The ID of the gpu")

    cli_args = parser.parse_args()
    dataset = SeamlessDataset(PROJECT_ROOT / cli_args.dataset_path)
    all_file_ids = dataset.file_ids
    #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = PROJECT_ROOT / "logs" / f"segmenter_{dataset.name}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="INFO")

    config_path = PROJECT_ROOT / cli_args.config 
    #
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Filter out any keys that have 'comment' in the name
    config_data = {k: v for k, v in config_data.items() if "comment" not in k.lower()}  

    if config_data.get("include_stage_2_partner_data", False):
        num_workers = 1
        logger.info("Stage 2 partner data enabled. Forcing single worker mode (workers=1) to preserve interaction pairs.")
    else:
        num_workers = cli_args.workers if cli_args.workers else 4
        logger.info(f"Initializing segmentation with {num_workers} workers on GPU {cli_args.gpu_id}")
    
    separated_files = np.array_split(all_file_ids, num_workers)

    processes = []
    for i in range(num_workers):
        # The workers share the same GPU
        process = mp.Process(
            target=run_worker, 
            args=(i if num_workers != 1 else -1, separated_files[i].tolist(), config_data, cli_args.dataset_path, cli_args.gpu_id)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    logger.info("All workers completed. Will now merge the LMDB folders in the cache directory.")

    if num_workers > 1:
        output_path = PROJECT_ROOT / "cache" / "segmenter" / config_data.get("project_name", "unknown") / dataset.name
        merge_parts(output_path, dataset.name, config_data)

        env = lmdb.open(str(output_path), readonly=True)
        with env.begin() as transaction:
            stats = env.stat()
            logger.info(f"Merged Dataset has: {stats['entries']} entries")
            cursor = transaction.cursor()
            if cursor.first():
                for key, value in cursor:
                    entry = pickle.loads(value)
                    logger.info(f"--- Key: {key.decode('ascii')} ---")
                    for field_name, field_value in entry.items():
                        if hasattr(field_value, "shape"):
                            logger.info(f"  {field_name} shape: {field_value.shape}")
                        else:
                            logger.info(f"  {field_name}: {field_value}")
                    break
            else:
                logger.error(f"Merged Dataset | Verification Failed: Database is empty/not found!")
    else:
        logger.info("Single worker detected. Bypassing merge phase.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()