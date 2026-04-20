import pandas as pd
import json
from pathlib import Path
from .seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS
import random
import argparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_assets_dir = str(PROJECT_ROOT / "data" / "seamless" / "assets")

class SeamlessDownloader:
    def __init__(self, download_name='unnamed'):
        """
        Initializes the downloader with pathing for raw data and cached filters.
        """
        self.filelist_path = _assets_dir + "/filelist.csv"
        self.output_dir = PROJECT_ROOT / "data" / "seamless" / "downloads" / f"{download_name}"
        self.cache_dir = PROJECT_ROOT / "cache" / "csv_filtering_cache"
        
        # make sure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.filelist_df = pd.read_csv(self.filelist_path)
        self.download_name = download_name

    def get_filtered_list(self, filters: dict):
        """
        Filters the dataframe and saves/loads a cached CSV named after the filter criteria.
        Example filters: {'has_annotation_1p': True, 'label': 'naturalistic', 'split': 'train'}
        """
        # create a filename based on the provided filter
        filter_str = "_".join([f"{k}-{v}" for k, v in sorted(filters.items())])
        cache_file = self.cache_dir / f"list_{filter_str}.csv"
        
        # See if the csv already exists based on the filename, if so we dont need to create it
        if cache_file.exists():
            print(f"Loading cached filter: {cache_file.name}")
            return pd.read_csv(cache_file), str(cache_file)

        # apply the filter to the main filelist, to get the filtered dataframe back
        filtered_df = self.filelist_df.copy()
        for column, value in filters.items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column] == value]
        
        # save to cache for future use
        if not filtered_df.empty:
            filtered_df.to_csv(cache_file, index=False)
            print(f"Created new filter cache: {cache_file.name} ({len(filtered_df)} items)")
        return filtered_df, str(cache_file)

    def download_subset(self, filters=None, csv_path=None, num_samples=None, preferred_vendors=True, random_draw=False, force_pairs=False, seed=42, file_list_only=False):
        """
        Downloads a batch of files based on a filter dictionary or a direct CSV path.
        Filters would be in the same notation as mentioned in def get_filtered_list
        """
        if csv_path:
            print(f"Using provided CSV path: {csv_path}")
            filtered_df = pd.read_csv(csv_path)
            active_csv = csv_path
        elif filters:
            filtered_df, active_csv = self.get_filtered_list(filters)
        else:
            print("Error: Either filters or csv_path must be provided.")
            return []
        
        if filtered_df.empty:
            print("Warning: No files found to download.")
            return []

        if file_list_only:
            return []

        # Configure the seamless connection using information from the first row of the df
        sample = filtered_df.iloc[0]
        config = DatasetConfig(
            label=sample['label'], 
            split=sample['split'], 
            preferred_vendors_only=preferred_vendors
        )
        
        fs = SeamlessInteractionFS(config=config, filelist_path=active_csv)

        if force_pairs:
            # Get the interaction key without the participant id to get pairs (Vxx_Sxxxx_Ixxxxxxxx)
            filtered_df['interaction_key'] = filtered_df['file_id'].str.rsplit('_', n=1).str[0]
            # Only keep interactions where both participants are present in the filtered list
            valid_pairs = filtered_df.groupby('interaction_key').filter(lambda x: len(x) == 2)
            selection_pool = valid_pairs['interaction_key'].unique().tolist()
        else:
            selection_pool = filtered_df['file_id'].tolist()

        if random_draw:
            if seed is not None:
                random.seed(seed)
                print(f"Applying random seed: {seed}")
            random.shuffle(selection_pool)
            print(f"Shuffling the selection pool.")

        if num_samples:
            selection_pool = selection_pool[:num_samples]

        if force_pairs:
            # Flatten the interaction pairs
            file_ids = filtered_df[filtered_df['interaction_key'].isin(selection_pool)]['file_id'].tolist()
        else:
            file_ids = selection_pool

        print(f"Starting download of {len(file_ids)} interactions to {self.output_dir}...")
        fs.download_batch_from_s3(file_ids, local_dir=self.output_dir)
        
        # Generate the meta data for this dataset
        meta_info = filters if filters else {"source_csv": Path(active_csv).name}
        self.generate_metadata(file_ids, meta_info)
        
        return file_ids

    def generate_metadata(self, file_ids, filters):
        """
        Saves a JSON manifest of what was downloaded for future reference
        """
        meta_path = self.output_dir / "download_manifest.json"
        manifest = {
            "download_name": self.download_name,
            "filter_criteria": filters,
            "downloaded_ids": file_ids,
            "count": len(file_ids),
            "status": "raw"
        }
        
        with open(meta_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        print(f"Manifest saved to {meta_path}")

def main():
    """Use this to setup your download, afterwards run this file to download"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_name", default="unknown", type=str, help="The name of this download")
    parser.add_argument("--filter_dict", default={}, type=str, help="JSON string of the filter dict")
    parser.add_argument("--preferred_vendors", default=True, type=bool, help="If we want to download from preferred vendors")
    parser.add_argument("--force_pairs", default=True, type=bool, help="If we want to force pairs, implying we always want the interaction partner downloaded too")
    parser.add_argument("--num_samples", default=True, type=int, help="the number of samples to download")
    parser.add_argument("--random_draw", default=True, type=bool, help="If we randomly draw")
    parser.add_argument("--seed", default=None, type=int, help="The seed, which is relevant if random_draw is True")
    parser.add_argument("--file_list_only", default=False, type=bool, help="If you only want the filelist to be created, and not commit to a full download")
    cli_args = parser.parse_args()

    filter_dict = json.loads(cli_args.filter_dict)

    downloader = SeamlessDownloader(download_name=cli_args.download_name)
    downloader.download_subset(filters=filter_dict, num_samples=cli_args.num_samples, random_draw=cli_args.random_draw, 
                               force_pairs=cli_args.force_pairs, preferred_vendors=cli_args.preferred_vendors,
                               seed=cli_args.seed,
                               file_list_only=cli_args.file_list_only)

if __name__ == "__main__":
    main()