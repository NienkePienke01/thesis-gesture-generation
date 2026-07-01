import os
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
import heapq
from torch.utils.data import DataLoader
from pathlib import Path
from data_processing.seamless_dataloader import SeamlessDataLoader
from .stage2_trainer_seamless import SeamlessStage2Trainer
import json
from types import SimpleNamespace
import argparse
from tqdm import tqdm
from datetime import datetime
from .train_ae_trainer_seamless import find_free_port, ddp_setup
from models_dyadit.vqvae_downsampled import TemporalVertexCodecDownsampled
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
checkpoint_dir = PROJECT_ROOT / "checkpoints_stage_2"

def train_model(rank, gpu_amount_size, shared_port, args):
    ddp_setup(rank, gpu_amount_size, shared_port)
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    local_checkpoint_dir = checkpoint_dir / args.project / start_time
    
    if rank == 0:
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(project="Seamless_Stage2_Training", name=f"{args.project}_stage2_run_{start_time}", config=vars(args))
    
    actual_gpu_id = args.gpus[rank]

    train_dataset = SeamlessDataLoader(args.data.train_lmdb, stage=2)
    train_sampler = DistributedSampler(train_dataset, num_replicas=gpu_amount_size, rank=rank)
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.training_params.batch_size, 
                            sampler=train_sampler, 
                            pin_memory=True,
                            num_workers=args.workers_per_gpu,
                            persistent_workers=True
                            )

    val_dataset = SeamlessDataLoader(args.data.val_lmdb, stage=2)
    validation_sampler = DistributedSampler(val_dataset, num_replicas=gpu_amount_size, rank=rank)
    validation_loader = DataLoader(val_dataset, 
                            batch_size=args.training_params.batch_size, 
                            sampler=validation_sampler, 
                            pin_memory=True,
                            num_workers=args.workers_per_gpu,
                            persistent_workers=True
                            )

    trainer = SeamlessStage2Trainer(args, actual_gpu_id)

    top_k_checkpoints = []
    global_epoch = 0
    total_steps = args.training_params.epochs * len(train_loader)

    for epoch in range(args.training_params.epochs):
        global_epoch = epoch
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        
        epoch_losses = {"total": 0.0, "ce": 0.0, "mse": 0.0, "ce_loss_layered": None, "masked_loss": 0.0}
        progress_bar = tqdm(train_loader, disable=(rank != 0))
        
        # Training Loop
        for batch_idx, batch_data in enumerate(progress_bar):
            current_step = (epoch * len(train_loader)) + batch_idx
            step_percentage = current_step / total_steps
            losses = trainer.train_step(batch_data, step_percentage)
            for k, v in losses.items():
                if k == "ce_loss_layered":
                    if epoch_losses[k] is None:
                        epoch_losses[k] = {layer: 0.0 for layer in v.keys()}
                    for layer, val in v.items():
                        epoch_losses[k][layer] += val
                else:
                    epoch_losses[k] += v
        
        # Validation Loop
        val_losses = {"total": 0.0, "ce": 0.0, "mse": 0.0, "ce_loss_layered": None}
        for batch_data in validation_loader:
            losses = trainer.val_step(batch_data)
            for k, v in losses.items():
                if k == "ce_loss_layered":
                    if val_losses[k] is None:
                        val_losses[k] = {layer: 0.0 for layer in v.keys()}
                    for layer, val in v.items():
                        val_losses[k][layer] += val
                else:
                    val_losses[k] += v
                
        trainer.scheduler.step()
            
        # Logging & Checkpoint
        if rank == 0 and (epoch % args.log_period == 0 or epoch == args.training_params.epochs - 1):
            avg_train_total = epoch_losses["total"] / len(train_loader)
            avg_val_total = val_losses["total"] / len(validation_loader)
            layered_metrics_train = {
                f"train/ce_layer_{k}": v / len(train_loader) 
                for k, v in epoch_losses["ce_loss_layered"].items()
            }
            layered_metrics_val = {
                f"val/ce_layer_{k}": v / len(validation_loader) 
                for k, v in val_losses["ce_loss_layered"].items()
            }

            wandb.log({
                "epoch": epoch,
                "train/total_loss": avg_train_total,
                "train/ce": epoch_losses["ce"] / len(train_loader),
                "train/mse": epoch_losses["mse"] / len(train_loader),
                "train/masked_ce": epoch_losses["masked_loss"] / len(train_loader),
                "val/total_loss": avg_val_total,
                "val/ce": val_losses.get("ce", 0) / len(validation_loader),
                "val/mse": val_losses.get("mse", 0) / len(validation_loader),
                "lr": trainer.optimizer.param_groups[0]['lr'],
                **layered_metrics_train,
                **layered_metrics_val
            })
            print(f"Epoch {epoch} | Train Loss: {avg_train_total:.4f} | Val Loss: {avg_val_total:.4f}")
            
            # Save top-k checkpoints (min-heap tracks the highest losses to remove them)
            save_path = local_checkpoint_dir / f"model_epoch{epoch}_loss{avg_val_total:.4f}.bin"
            model_state = trainer.model.module.state_dict() if gpu_amount_size > 1 else trainer.model.state_dict()
            
            if len(top_k_checkpoints) < 3:
                torch.save(model_state, save_path)
                heapq.heappush(top_k_checkpoints, (-avg_val_total, save_path))
            elif avg_val_total < -top_k_checkpoints[0][0]:
                _, old_path = heapq.heappop(top_k_checkpoints)
                if os.path.exists(old_path): 
                    os.remove(old_path)
                torch.save(model_state, save_path)
                heapq.heappush(top_k_checkpoints, (-avg_val_total, save_path))
            
            # Periodic Backups
            if epoch % args.periodic_backup_after_x_epochs == 0:
                torch.save(model_state, local_checkpoint_dir / f"periodic_backup_epoch_{epoch}.bin")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': avg_val_total
                }, local_checkpoint_dir / f"periodic_backup_complete_{epoch}.bin")

    if rank == 0:
        model_state = trainer.model.module.state_dict() if gpu_amount_size > 1 else trainer.model.state_dict()
        final_ckpt_path = local_checkpoint_dir / f"stage2_final_epoch_{global_epoch}.bin"
        torch.save(model_state, final_ckpt_path)

        # Only run evaluation, if there is evaluation config present in the main stage 2 config
        if hasattr(args, "evaluation"):
            print("\n" + "="*50)
            print("STARTING AUTOMATED EVALUATION PIPELINE")
            print("="*50)
            
            # Nuke the trainer and clear VRAM to prevent CUDA out of memory
            del trainer
            torch.cuda.empty_cache()
            
            eval_args = args.evaluation
            chunks = eval_args.chunk_frames if isinstance(eval_args.chunk_frames, list) else [eval_args.chunk_frames]
            
            # Generate one timestamp for this entire evaluation run
            eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for chunk in chunks:
                print(f"\n>>> Running Inference for chunk size: {chunk} frames")
                
                # Set the paths we want the child scripts to use
                explicit_lmdb_path = PROJECT_ROOT / "evaluation_results" / f"eval_auto_{args.project}_chunk_{chunk}_{eval_timestamp}"
                explicit_metrics_path = PROJECT_ROOT / "evaluation_metrics_results" / f"metrics_auto_{args.project}_chunk_{chunk}_{eval_timestamp}.json"
                explicit_video_render_path = PROJECT_ROOT / "qualitative_renders" / f"eval_auto_{args.project}_{eval_timestamp}"
                infer_cmd_prefix = getattr(eval_args, "infer_cmd_prefix", ["python"])

                # Run Inference
                infer_cmd = infer_cmd_prefix + [
                    "--stage2_config", args.config_path,
                    "--stage2_checkpoint", str(final_ckpt_path),
                    "--segmenter_config", eval_args.segmenter_config,
                    "--test_folder", eval_args.test_folder,
                    "--chunk_frames", str(chunk),
                    "--max_gb", str(eval_args.max_gb),
                    "--device", "cuda",
                    "--output_dir", str(explicit_lmdb_path)
                ]
                subprocess.run(infer_cmd, check=True)
                
                print(f"\n>>> Running Evaluation on {explicit_lmdb_path.name}")
                
                eval_cmd_prefix = getattr(eval_args, "eval_cmd_prefix", ["python"])
                # Run Evaluation
                eval_cmd = eval_cmd_prefix + [
                    "--results", str(explicit_lmdb_path),
                    "--metrics_output", str(explicit_metrics_path)
                ]
                subprocess.run(eval_cmd, check=True)

                # Parse and Log to WandB
                if explicit_metrics_path.exists():
                    with open(explicit_metrics_path, 'r') as f:
                        metrics = json.load(f)
                    
                    wandb_log_dict = {f"eval_{chunk}_frames/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                    wandb.log(wandb_log_dict)
                    print(f"Successfully logged metrics for {chunk} frames to WandB!")
                else:
                    print(f"Error: Expected metrics file at {explicit_metrics_path} but it was not found.")

            print(f"\n>>> Generating videos for {explicit_video_render_path.name}")

            infer_videos_prefix = getattr(eval_args, "infer_videos_prefix", ["python"])
            infer_videos_cmd = infer_videos_prefix + [
                "--stage2_config", args.config_path,
                "--stage2_checkpoint", str(final_ckpt_path),
                "--segmenter_config", eval_args.segmenter_config,
                "--data_folder", eval_args.test_folder,
                "--output_dir", str(explicit_video_render_path),
                "--warm_start",
                "--file_ids_txt", "evaluation/file_list_videos.txt",
                "--device", "cuda"
            ]
            subprocess.run(infer_videos_cmd, check=True)

        wandb.finish()

    destroy_process_group()

def main():
    shared_port = find_free_port()
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2_model.json", help="Path to the training config.json")
    cli_args = parser.parse_args()
    
    config_path = PROJECT_ROOT / cli_args.config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Convert dictionary blocks to SimpleNamespaces
    args = SimpleNamespace(**config_dict)
    args.data = SimpleNamespace(**config_dict["data"])
    args.training_params = SimpleNamespace(**config_dict["training_params"])
    args.loss_weights = SimpleNamespace(**config_dict["loss_weights"])
    
    if "stage_1_pretrain" in config_dict:
        args.stage_1_pretrain = SimpleNamespace(**config_dict["stage_1_pretrain"])
    else:
        args.stage_1_pretrain = None

    if "evaluation" in config_dict:
        args.evaluation = SimpleNamespace(**config_dict["evaluation"])

    args.model_params = SimpleNamespace(**config_dict["model_params"])

    gpu_amount_size = len(args.gpus)
    if gpu_amount_size > 1:
        print(f"Running with {gpu_amount_size} gpus")
        mp.spawn(train_model, args=(gpu_amount_size, shared_port, args), nprocs=gpu_amount_size)
    else:
        print("Running with 1 gpu")
        train_model(0, 1, shared_port, args)

if __name__ == "__main__":
    main()