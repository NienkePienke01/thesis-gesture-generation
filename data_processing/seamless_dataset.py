import json
import os
from pathlib import Path
from .utils import SeamlessSMPLHLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class SeamlessDataset:
    def __init__(self, data_root: str):
        """
        Initializes a sub seamless interaction dataset by reading the manifest and mapping file IDs to paths.
        data_root: The path to the that the dataset is in.
        If this was downloaded with SeamlessDownloader it is the folder that contains the .json manifest
        """
        self.data_root = Path(data_root)
        self.manifest_path = self.data_root / "download_manifest.json"
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.filter_criteria = self.manifest.get("filter_criteria", {})
        self.file_ids = self.manifest.get("downloaded_ids", [])
        
        # Build the lookup table for easy access
        self.samples = self._build_path_map()
        
        self.name = self.manifest.get("download_name", "unknown_name")

    def _build_path_map(self):
        """
        Scans the split folder once to map IDs to their actual disk locations.
        """
        # path based on manifest (./data/naturalistic/train)
        base_search_path = (self.data_root / self.filter_criteria.get("label", "") / self.filter_criteria.get("split", ""))
        
        print(f"Indexing .npz files in {base_search_path}...")
        
        # map filename stem to full Path
        disk_map = {path.stem: str(path.absolute()) for path in base_search_path.rglob("*.npz")}
        
        # return only the subset mentioned in the manifest
        final_map = {file_id: disk_map[file_id] for file_id in self.file_ids if file_id in disk_map}
        
        print(f"Found {len(final_map)} files out of {self.manifest.get('count', 'Unknown')} successfully.")
        return final_map

    def get_interaction(self, index: int, loader_args: dict):
        """
        Returns a SeamlessSMPLHLoader for a specific sample.
        """
        return SeamlessSMPLHLoader(self.samples[index], **loader_args)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"SeamlessDataset(count={len(self.samples)}, criteria={self.filter_criteria})"

def main():
    dataset = SeamlessDataset(PROJECT_ROOT / "data" / "seamless" / "downloads" / "experimenting")
    print(dataset.get_interaction(dataset.file_ids[0], {'load_vertices': False, 'with_meta_data': True}))

if __name__ == "__main__":
    main()