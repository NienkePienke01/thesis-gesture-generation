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
                "pose": torch.from_numpy(data["pose"]).float(),
                "raw_audio": torch.from_numpy(data["raw_audio"]).float(),
                "wav2vec_audio": torch.from_numpy(data["wav2vec_audio"]).float(),
                "hubert_audio": torch.from_numpy(data["hubert_audio"]).float(),
                "bert": torch.from_numpy(data["bert"]).float(),
                "full_id": data["seamless_id"],
                "semantic_scores": torch.from_numpy(data["semantic_scores"]).float(),
                "partner_raw_audio": torch.from_numpy(data["partner_raw_audio"]).float(),
                "partner_wav2vec_audio": torch.from_numpy(data["partner_wav2vec_audio"]).float(),
                "partner_hubert_audio": torch.from_numpy(data["partner_hubert_audio"]).float(),
                "partner_pose": torch.from_numpy(data["partner_pose"]).float(),
                "partner_bert": torch.from_numpy(data["partner_bert"]).float(),
                "partner_full_id": data["partner_seamless_id"],
                "conversation_states": torch.from_numpy(data["conversation_states"]).long(),
                "partner_conversation_states": torch.from_numpy(data["partner_conversation_states"]).long(),
                "social_relationship": torch.tensor(data["social_relationship"], dtype=torch.long),
                "personality_traits": torch.from_numpy(data["personality_traits"]).float(),
                "partner_personality_traits": torch.from_numpy(data["partner_personality_traits"]).float(),
                "partner_emotions": torch.from_numpy(data["partner_emotions"]).int(),
                "emotions": torch.from_numpy(data["emotions"]).int(),
            }

        return output