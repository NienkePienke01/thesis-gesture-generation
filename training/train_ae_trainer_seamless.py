import os
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
import heapq
from .ae_trainer_seamless import SeamlessStage1Trainer
from torch.utils.data import DataLoader
from .ae_trainer_seamless import SeamlessStage1Trainer
from pathlib import Path
from data_processing.seamless_dataloader import SeamlessDataLoader
import json
from types import SimpleNamespace
import argparse
from tqdm import tqdm
import socket
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
checkpoint_dir = PROJECT_ROOT / "checkpoints"

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def ddp_setup(rank, gpu_amount_size, shared_port):
    print(f"Setting up worker for rank: {rank}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(shared_port)
    init_process_group(backend="nccl", rank=rank, world_size=gpu_amount_size)
    print(f"Setup of rank: {rank} done")

def train_model(rank, gpu_amount_size, shared_port, args):
    """
    train_dataset_path and validation_dataset_path should be paths to a lmdb file
    """
    ddp_setup(rank, gpu_amount_size, shared_port)
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    local_checkpoint_dir = checkpoint_dir / args.project / start_time
    if not local_checkpoint_dir.exists():
        local_checkpoint_dir.mkdir(parents=True)

    if rank == 0:
        wandb.init(project="Seamless_Training", name=f"{args.project}_{args.mode}_ddp_run_{start_time}", config=vars(args))
    
    actual_gpu_id = args.gpus[rank]
    
    print("Setting up data loaders...")
    train_dataset = SeamlessDataLoader(args.data.train_lmdb, stage=1)
    train_sampler = DistributedSampler(train_dataset, num_replicas=gpu_amount_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.training_params.batch_size, sampler=train_sampler, pin_memory=True)

    val_dataset = SeamlessDataLoader(args.data.val_lmdb, stage=1)
    validation_loader = DataLoader(val_dataset, batch_size=args.training_params.batch_size, pin_memory=True)
    
    stats_path = Path(args.data.train_lmdb).parent / "mean_and_std.pt"
    args.stats_path = stats_path
    trainer = SeamlessStage1Trainer(args, actual_gpu_id)

    best_validation_loss = float('inf')
    # training loop
    top_k_checkpoints = []
    print("Starting training loop...")
    global_epoch = 0
    for epoch in range(args.training_params.epochs):
        global_epoch = epoch
        train_sampler.set_epoch(epoch)
        
        epoch_losses = {}
        progress_bar = tqdm(train_loader, disable=(rank != 0))
        # Training loss
        for batch_data in progress_bar:
            losses = trainer.train_step(batch_data)
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
        
        # Validation loss
        val_losses = trainer.val_step(next(iter(validation_loader)))
        avg_validation_loss = val_losses["total"]
        
        if rank == 0 and (epoch % args.log_period == 0 or epoch == args.training_params.epochs - 1):
            avg_train_loss = epoch_losses["total"] / len(train_loader)

            wandb.log({
                "epoch": epoch,
                "train/total_loss": avg_train_loss,
                "val/total_loss": avg_validation_loss,
                "val/rec": val_losses.get("rec", 0),
                "val/ver": val_losses.get("ver", 0),
                "val/vel": val_losses.get("vel", 0),
                "val/acc": val_losses.get("acc", 0),
                "val/commit": val_losses.get("commit", 0),
                "val/perplexity": val_losses.get("perplexity", 0),
                "lr": trainer.opt.param_groups[0]['lr'],
                **{f"train/{k}": v / len(train_loader) for k, v in epoch_losses.items() if k != "total"}
            })
            
            # save the best top-k checkpoints
            save_path = local_checkpoint_dir / f"model_epoch{epoch}_loss{avg_validation_loss:.4f}.bin"
            if len(top_k_checkpoints) < 3:
                torch.save(trainer.model.module.state_dict(), save_path)
                heapq.heappush(top_k_checkpoints, (-avg_validation_loss, save_path))
            elif avg_validation_loss < -top_k_checkpoints[0][0]:
                _, old_path = heapq.heappop(top_k_checkpoints)
                if os.path.exists(old_path): os.remove(old_path)
                torch.save(trainer.model.module.state_dict(), save_path)
                heapq.heappush(top_k_checkpoints, (-avg_validation_loss, save_path))
            
            # Periodic Backup
            if epoch % args.periodic_backup_after_x_epochs == 0:
                backup_path = local_checkpoint_dir / f"periodic_backup_epoch_{epoch}.bin"
                complete_backup_path = local_checkpoint_dir / f"periodic_backup_complete_{epoch}.bin"
                torch.save(trainer.model.module.state_dict(), backup_path)
                torch.save({'epoch': epoch,
                            'model_state_dict': trainer.model.module.state_dict(),
                            'optimizer_state_dict': trainer.opt.state_dict(),
                            'scheduler_state_dict': trainer.scheduler.state_dict(),
                            'loss': avg_validation_loss}, complete_backup_path)

        trainer.scheduler.step(avg_validation_loss)
    
    if rank == 0:
        final_save_path = local_checkpoint_dir / f"model_final_epoch_{global_epoch}_val_loss_{avg_validation_loss}.bin"
        torch.save(trainer.model.module.state_dict(), final_save_path)
        print(f"Final model saved to {final_save_path}")
        wandb.finish()

    destroy_process_group()
    
    print("Finished training...")

def main():
    shared_port = find_free_port()
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/seamless_stage_1_upper_training.json", help="Path to the training config.json")
    cli_args = parser.parse_args()
    
    config_path = PROJECT_ROOT / cli_args.config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    args = SimpleNamespace(**config_dict)
    args.data = SimpleNamespace(**config_dict["data"])
    args.training_params = SimpleNamespace(**config_dict["training_params"])
    args.loss_weights = SimpleNamespace(**config_dict["loss_weights"])
    args.model_params = config_dict["model_params"]

    gpu_amount_size = len(args.gpus)
    if gpu_amount_size > 1:
        print(f"Running with {gpu_amount_size} gpus")
        mp.spawn(train_model, args=(gpu_amount_size, shared_port, args), nprocs=gpu_amount_size)
    else:
        print("Running with 1 gpu")
        train_model(0, 1, shared_port, args)

if __name__ == "__main__":
    main()