import lmdb
import pickle
import torch
import librosa
import numpy as np
import math
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from .seamless_dataset import SeamlessDataset
from .utils import SeamlessSMPLHLoader
from transformers import AutoTokenizer, BertModel
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cache_dir = PROJECT_ROOT / "cache" / "segmenter"

class SeamlessSegmenter:
    def __init__(self, dataset : SeamlessDataset, pose_fps=30, audio_sr=16000, pose_length=64, stride=10, seed=42, max_gb=10,
                 clean_first_seconds=0, clean_final_seconds=0):
        self.dataset = dataset
        self.output_path = cache_dir / f"{self.dataset.name}"
        self.pose_fps = pose_fps
        self.audio_sr = audio_sr
        self.pose_length = pose_length
        self.stride = stride
        self.max_gb = max_gb

        self.clean_first = clean_first_seconds
        self.clean_final = clean_final_seconds

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").eval()

    def build_cache(self):
        """Processes all interactions in a given seamless interaction dataset and writes the segments to LMDB."""
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

        # create/reserve the lmdb cache
        env = lmdb.open(str(self.output_path), map_size=int(1024**3 * self.max_gb))
        
        sample_idx = 0
        interaction_idx = 0
        for _, _ in tqdm(self.dataset.samples.items(), desc="segmenting dataset"):
            # Get a seamless loader, of the data sample for easy access to all needed information
            loader = self.dataset.get_interaction(self.dataset.file_ids[interaction_idx], {'load_vertices': False, 'with_meta_data': True})
            
            full_pose = loader.get_trainable_pose() # (frames, 129)
            
            # load and resample Audio
            audio_data, _ = librosa.load(loader.meta_data.audio_file, sr=self.audio_sr)

            # get the lookup table for the bert word vectors
            bert_lookup_table = self._get_bert_transcript(loader.meta_data.transcript)
            # get the word vectors list len(frames) containing for each index {'word': word, 'sentence': sentence, 'embedding': vector}
            word_list = self._get_word_list(loader, bert_lookup_table)
            # get the emotion category for each frame
            emotion_category_frames = self._get_frame_emotions(loader)
            # get the semantic value for each frame
            semantic_values_list = self._get_frame_semantics(loader)
            
            # align the audio and pose
            aligned_pose, aligned_audio, aligned_words, aligned_emotion_list, aligned_semantic_values= self._align_data(full_pose, audio_data, word_list, emotion_category_frames, semantic_values_list)

            # see how many 64-frame chunks fit
            num_subdivisions = math.floor((len(aligned_pose) - self.pose_length) / self.stride) + 1
            audio_samples_per_frame = self.audio_sr / self.pose_fps # how many audio samples represent the duration of one motion frame
            audio_chunk_len = int(self.pose_length * audio_samples_per_frame) # chunk sizes for the audio compared

            with env.begin(write=True) as transaction:
                for i in range(num_subdivisions):
                    # motion segment
                    pose_start = i * self.stride
                    pose_end = pose_start + self.pose_length
                    sample_pose = aligned_pose[pose_start:pose_end]
                    
                    # audio segment
                    audio_start = int(pose_start * audio_samples_per_frame)
                    audio_end = audio_start + audio_chunk_len
                    sample_audio = aligned_audio[audio_start:audio_end]

                    # word segment
                    sample_words = aligned_words[pose_start:pose_end]
                    sample_bert = np.array([f["embedding"] for f in sample_words])
                    sample_word_text = np.array([f["word"] for f in sample_words])
                    sample_sentence = np.array([f["sentence"] for f in sample_words])

                    # emotion segment
                    sample_emotions = aligned_emotion_list[pose_start:pose_end]

                    # semantic values segment
                    sample_semantic_values = aligned_semantic_values[pose_start:pose_end]

                    # storage setup
                    entry = {
                        "pose": sample_pose.numpy(), # pose segment of 129 dimension
                        "audio": sample_audio, # audio segment
                        "bert": sample_bert, # word segments
                        "emotions": sample_emotions, # placeholder emo
                        "semantic_scores": sample_semantic_values, # semantic values for each frame being 0.0, 0.1 or 1.0
                        "sample_word": sample_word_text, # the actual words in text for each frame of the sample
                        "sample_sentence": sample_sentence, # the sentence for each frame, where the word came from
                        "segment_index": i, # the index of this segment for this specific file
                        "seamless_id": loader.meta_data.full_id, # video/file ID
                        "file_path": loader.path, # the path this file came from
                    }
                    
                    key = f"{sample_idx:08d}".encode("ascii")
                    serialized_entry = pickle.dumps(entry)
                    transaction.put(key, serialized_entry)
                    sample_idx += 1
            interaction_idx +=1
        env.close()
        logger.info(f"Finished. Total segments cached: {sample_idx}")

    def _align_data(self, full_pose, audio_data, word_list, emotion_list, semantic_values_list):
        """
        Aligns the motion .npz and audio .wav by finding the shortest common duration 
        """
        round_seconds_pose = full_pose.shape[0] // self.pose_fps
        round_seconds_audio = len(audio_data) // self.audio_sr

        round_seconds_common = min(round_seconds_pose, round_seconds_audio)
        
        # (skip first/last x amount of seconds)
        clipose_start_time = self.clean_first
        clipose_end_time = round_seconds_common - self.clean_final
        
        # pose indices
        pose_start = clipose_start_time * self.pose_fps
        pose_end = clipose_end_time * self.pose_fps
        
        # audio indices
        audio_start = clipose_start_time * self.audio_sr
        audio_end = clipose_end_time * self.audio_sr
        
        # slice the arrays to the aligned and cleaned window
        aligned_pose = full_pose[pose_start:pose_end]
        aligned_word_list = word_list[pose_start:pose_end]
        aligned_emotion_list = emotion_list[pose_start:pose_end]
        aligned_audio = audio_data[audio_start:audio_end]
        aligned_semantic_values = semantic_values_list[pose_start:pose_end]
        
        return aligned_pose, aligned_audio, aligned_word_list, aligned_emotion_list, aligned_semantic_values
    
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
        return np.argmax(emotion_scores, axis=1).astype(np.int32) if emotion_scores is not None else np.zeros(loader.num_frames, dtype=np.float32)

    def _get_frame_semantics(self, loader : SeamlessSMPLHLoader):
        """
        returns semantic values for each frame
        values cane be 
        0.0 for no speaking frames
        0.1 for speaking frames
        1.0 for any motion of interest frames
        """
        vad_info = loader.meta_data.transcript.get('metadata:vad', None)
        # initialise for 0.0 silence
        sem_scores = np.zeros(loader.num_frames, dtype=np.float32)
        if vad_info != None:
            for vad_segment in vad_info:
                start_frame = int(vad_segment['start'] * self.pose_fps)
                end_frame = int(vad_segment['end'] * self.pose_fps)
                # mark it as speech (0.1)
                sem_scores[max(0, start_frame):min(loader.num_frames, end_frame)] = 0.1

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

def main():
    parser = argparse.ArgumentParser(description="Seamless Segmenter Parser")
    parser.add_argument("--dataset_path", type=str, default="data/seamless/downloads/experimenting", help="Path to the dataset folder")
    parser.add_argument("--config", type=str, default="configs/seamless_segmenter.json", help="Path to the segmenter config.json")
    cli_args = parser.parse_args()
    dataset = SeamlessDataset(PROJECT_ROOT / cli_args.dataset_path)

    config_path = PROJECT_ROOT / cli_args.config 
    #
    with open(config_path, "r") as f:
        config_data = json.load(f)
    segmenter = SeamlessSegmenter(dataset, **config_data)
    segmenter.build_cache()

    # short check to see if it all worked fine
    cache_path = str(PROJECT_ROOT / "cache" / "segmenter" / f"{dataset.name}")
    env = lmdb.open(cache_path, readonly=True)
    with env.begin() as transaction:
        raw = transaction.get(b"00000000")
        entry = pickle.loads(raw)
        print(f"Pose shape: {entry['pose'].shape}") # should be (64, 129)
        print(f"Audio shape: {entry['audio'].shape}")
        print(f"BERT shape: {entry['bert'].shape}") # should be (64, 768)
        print(f"Emotions shape: {entry['emotions'].shape}")
        print(f"Semantics shape: {entry['semantic_scores'].shape}")

if __name__ == "__main__":
    main()