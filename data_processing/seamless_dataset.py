import json
import os
from pathlib import Path
from .utils import SeamlessSMPLHLoader
import numpy as np
import cv2
import random

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
    
    def get_total_hours(self):
        """
        Returns the total raw hours this dataset contains
        """
        total_frames = 0
        for path in self.samples.values():
            data = np.load(path, allow_pickle=True)
            total_frames += data['smplh:global_orient'].shape[0]
            
        total_seconds = total_frames / 30
        return round(total_seconds / 3600, 2)

    def get_screenshots_for_visual_evaluation(self):
        """
        Creates a folder, that contains 1 screenshot for each sample. This can be used for easy quick human visual evaluation of the quality of the data
        """
        folder_location = self.data_root / "screenshots"
        folder_location.mkdir(parents=True, exist_ok=True)
        for id in self.file_ids:
            loader = self.get_interaction(id, {'load_vertices': False, 'with_meta_data': True})
            video_path = loader.meta_data.video_file
            if video_path != None:
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                random_frame = random.randint(0, total_frames - 1)

                cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                success, frame = cap.read()
                
                if success:
                    screenshot_path = folder_location / f"{id}.png"
                    cv2.imwrite(str(screenshot_path), frame)
                cap.release()


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"SeamlessDataset(count={len(self.samples)}, criteria={self.filter_criteria})"

def main():
    """
    Used to do quick testing of this script
    """
    dataset = SeamlessDataset(PROJECT_ROOT / "data" / "seamless" / "downloads" / "annotation_observation")
    print(f"The dataset is a total of {dataset.get_total_hours()} raw hours")
    dataset.get_screenshots_for_visual_evaluation()
    # print(dataset.get_interaction(dataset.file_ids[0], {'load_vertices': False, 'with_meta_data': True}))

if __name__ == "__main__":
    main()