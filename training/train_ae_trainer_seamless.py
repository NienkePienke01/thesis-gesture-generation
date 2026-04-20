import torch
from torch.utils.data import DataLoader
from .ae_trainer_seamless import SeamlessStage1Trainer
from pathlib import Path
from data_processing.seamless_dataloader import SeamlessDataLoader
import json
from types import SimpleNamespace
import argparse
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
checkpoint_dir = PROJECT_ROOT / "checkpoints"

def train_model(args):
    """
    train_dataset_path and validation_dataset_path should be paths to a lmdb file
    """
    trainer = SeamlessStage1Trainer(args)
    
    print("Setting up data loaders...")
    train_dataset = SeamlessDataLoader(args.data.train_lmdb, stage=1)
    train_loader = DataLoader(train_dataset, batch_size=args.training_params.batch_size, shuffle=True)
    validation_dataset = SeamlessDataLoader(args.data.val_lmdb, stage=1)
    validation_loader = DataLoader(validation_dataset, batch_size=args.training_params.batch_size, shuffle=False)
    
    best_validation_loss = float('inf')
    # training loop
    print("Starting training loop...")
    for epoch in range(args.training_params.epochs):
        train_log_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for _, batch_data in enumerate(progress_bar):
            loss = trainer.train_step(batch_data)
            train_log_loss += loss
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        # validation
        validation_log_loss = 0
        for batch_data in validation_loader:
            validation_log_loss += trainer.val_step(batch_data)
            
        avg_train_loss = train_log_loss / len(train_loader)
        avg_validation_loss = validation_log_loss / len(validation_loader)
        
        print(f"Epoch {epoch+1} out of {args.training_params.epochs} Complete | Avg Train: {avg_train_loss:.4f} | Avg Val: {avg_validation_loss:.4f}")
        
        # save the best checkpoint
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            save_path = str(checkpoint_dir / f"{args.project}_{args.mode}_best.bin")
            torch.save(trainer.model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

        trainer.scheduler.step()
    
    print("Finished training...")

def main():
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
    train_model(args)

if __name__ == "__main__":
    main()