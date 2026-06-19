import torch
import lmdb
import pickle
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class SeamlessDataLoader(Dataset):
    def __init__(self, lmdb_path, stage=1):
        self.lmdb_path = str(PROJECT_ROOT / lmdb_path)
        self.env = None

        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.length = temp_env.stat()['entries'] 
        temp_env.close()
        self.stage = stage

    def _init_database(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._init_database()
        with self.env.begin(write=False) as transaction:
            key = f"{index:08d}".encode("ascii")
            byte_data = transaction.get(key)
            if byte_data is None:
                raise IndexError(f"Index {index} not found in LMDB.")
            data = pickle.loads(byte_data)
        
        if self.stage == 1:
            output = {
                "pose": torch.from_numpy(data["pose"]).float()
            }
        else:
            output = {
                "pose": torch.from_numpy(data["pose"]).float() if "pose" in data else None,
                "raw_audio": torch.from_numpy(data["raw_audio"]).float() if "raw_audio" in data else None,
                "wav2vec_audio": torch.from_numpy(data["wav2vec_audio"]).float() if "wav2vec_audio" in data else None,
                "hubert_audio": torch.from_numpy(data["hubert_audio"]).float() if "hubert_audio" in data else None,
                "bert": torch.from_numpy(data["bert"]).float() if "bert" in data else None,
                "full_id": data.get("seamless_id"),
                "semantic_scores": torch.from_numpy(data["semantic_scores"]).float() if "semantic_scores" in data else None,
                "partner_raw_audio": torch.from_numpy(data["partner_raw_audio"]).float() if "partner_raw_audio" in data else None,
                "partner_wav2vec_audio": torch.from_numpy(data["partner_wav2vec_audio"]).float() if "partner_wav2vec_audio" in data else None,
                "partner_hubert_audio": torch.from_numpy(data["partner_hubert_audio"]).float() if "partner_hubert_audio" in data else None,
                "partner_pose": torch.from_numpy(data["partner_pose"]).float() if "partner_pose" in data else None,
                "partner_bert": torch.from_numpy(data["partner_bert"]).float() if "partner_bert" in data else None,
                "partner_full_id": data.get("partner_seamless_id"),
                "conversation_states": torch.from_numpy(data["conversation_states"]).long() if "conversation_states" in data else None,
                "partner_conversation_states": torch.from_numpy(data["partner_conversation_states"]).long() if "partner_conversation_states" in data else None,
                "social_relationship": torch.tensor(data["social_relationship"], dtype=torch.long) if "social_relationship" in data else None,
                "personality_traits": torch.from_numpy(data["personality_traits"]).float() if "personality_traits" in data else None,
                "partner_personality_traits": torch.from_numpy(data["partner_personality_traits"]).float() if "partner_personality_traits" in data else None,
                "partner_emotions": torch.from_numpy(data["partner_emotions"]).int() if "partner_emotions" in data else None,
                "emotions": torch.from_numpy(data["emotions"]).int() if "emotions" in data else None,
            }

        return output