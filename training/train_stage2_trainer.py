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

    # Load validation data only on rank 0
    val_dataset = SeamlessDataLoader(args.data.val_lmdb, stage=2)
    validation_loader = DataLoader(val_dataset, batch_size=args.training_params.batch_size, pin_memory=True)

    trainer = SeamlessStage2Trainer(args, actual_gpu_id)

    top_k_checkpoints = []
    global_epoch = 0

    for epoch in range(args.training_params.epochs):
        global_epoch = epoch
        train_sampler.set_epoch(epoch)
        
        epoch_losses = {"total": 0.0, "motion_loss": 0.0, "state_loss": 0.0}
        progress_bar = tqdm(train_loader, disable=(rank != 0))
        
        # Training Loop
        for batch_data in progress_bar:
            losses = trainer.train_step(batch_data)
            for k, v in losses.items():
                epoch_losses[k] += v
        
        # Validation Loop
        val_losses = trainer.val_step(next(iter(validation_loader)))
                
        trainer.scheduler.step()
            
        # Metrics Logging & Checkpoint Management
        if rank == 0 and (epoch % args.log_period == 0 or epoch == args.training_params.epochs - 1):
            avg_train_total = epoch_losses["total"] / len(train_loader)
            avg_val_total = val_losses.get("total", 0)

            wandb.log({
                "epoch": epoch,
                "train/total_loss": avg_train_total,
                "train/motion_loss": epoch_losses["motion_loss"] / len(train_loader),
                "train/state_loss": epoch_losses["state_loss"] / len(train_loader),
                "val/total_loss": avg_val_total,
                "val/motion_loss": val_losses.get("motion_loss", 0),
                "val/state_loss": val_losses.get("state_loss", 0),
                "lr": trainer.opt.param_groups[0]['lr']
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
                    'optimizer_state_dict': trainer.opt.state_dict(),
                    'loss': avg_val_total
                }, local_checkpoint_dir / f"periodic_backup_complete_{epoch}.bin")

    if rank == 0:
        model_state = trainer.model.module.state_dict() if gpu_amount_size > 1 else trainer.model.state_dict()
        torch.save(model_state, local_checkpoint_dir / f"stage2_final_epoch_{global_epoch}.bin")
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