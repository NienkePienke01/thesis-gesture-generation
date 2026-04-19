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
        
        output = {
            "pose": torch.from_numpy(data["pose"]).float()
        }

        # Later i will add stage 2 output here

        return output