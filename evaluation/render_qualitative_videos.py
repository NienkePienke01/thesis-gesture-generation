import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# Import from your existing project structure
from data_processing.seamless_dataset import SeamlessDataset
from visualisation.visualise_stage_2 import load_stage2_and_stage1_pipeline, reconstruct_and_render

def dummy_progress(val, desc=""):
    """
    A dummy callable to override Gradio's gr.Progress() in the rendering functions.
    This keeps the console output clean while tqdm handles the overall progress bar.
    """
    pass

def main(args):
    # 1. Gather file IDs
    file_ids = []
    if args.file_ids:
        file_ids.extend(args.file_ids)
    if args.file_ids_txt:
        with open(args.file_ids_txt, 'r') as f:
            file_ids.extend([line.strip() for line in f if line.strip()])
            
    if not file_ids:
        print("Error: No file IDs provided! Use --file_ids or --file_ids_txt.")
        return

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Loading dataset from {args.data_folder}...")
    dataset = SeamlessDataset(args.data_folder, file_ids=file_ids)
    print(f"Successfully mapped {len(dataset.samples)} out of {len(file_ids)} requested IDs.")

    # Initialize the Pipeline
    print("\n>>> Initializing Stage 2 and Stage 1 Pipeline...")
    status = load_stage2_and_stage1_pipeline(
        args.stage2_config, 
        args.stage2_checkpoint, 
        args.segmenter_config, 
        args.device
    )
    print(status)

    # Inference & Rendering Loop
    print("\n" + "="*50)
    print("STARTING BATCH VIDEO RENDERING")
    print("="*50)
    
    success_count = 0
    for file_id in tqdm(dataset.file_ids, desc="Rendering Videos"):
        npz_path = str(dataset.samples.get(file_id))
        
        if not npz_path or not os.path.exists(npz_path):
            print(f"\nWarning: {file_id} not found on disk. Skipping.")
            continue
            
        # We pass dummy_progress to override the default gr.Progress() behavior
        try:
            video_path, msg = reconstruct_and_render(
                target_npz_file=npz_path, 
                device=args.device, 
                top_k=args.top_k, 
                top_p=args.top_p, 
                temperature=args.temperature, 
                warm_start=args.warm_start, 
                progress=dummy_progress
            )
            
            # Check if video was successfully generated
            if isinstance(video_path, str) and os.path.exists(video_path):
                # The video is saved in the cache folder initially. Copy it to our clean output folder.
                final_dest = output_dir / f"{file_id}_comparison.mp4"
                shutil.copy(video_path, final_dest)
                success_count += 1
            else:
                print(f"\nFailed to generate video for {file_id}. Message: {msg}")
                
        except Exception as e:
            print(f"\nException occurred while processing {file_id}: {str(e)}")

    print("\n" + "="*50)
    print(f"FINISHED! Successfully rendered {success_count}/{len(dataset.samples)} videos.")
    print(f"Videos saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render qualitative comparison videos.")
    
    # Path Arguments
    parser.add_argument("--stage2_config", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--segmenter_config", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True, help="Root folder of the Seamless dataset")
    parser.add_argument("--output_dir", type=str, default="qualitative_renders", help="Where to save the final MP4s")
    
    # Target IDs
    parser.add_argument("--file_ids", type=str, nargs='+', help="List of specific file IDs to render (space separated)")
    parser.add_argument("--file_ids_txt", type=str, help="Path to a .txt file containing one file ID per line")
    
    # Model Generation Params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--warm_start", action="store_true")
    
    main(parser.parse_args())