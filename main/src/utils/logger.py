import os
import torch
from datetime import datetime

def create_logger():
    """Create log and checkpoint folders + filenames."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"outputs/logs/train_{timestamp}.log"
    best_model_path = "outputs/checkpoints/best_model.pt"
    return log_file, best_model_path


def log_to_file(log_file, text):
    """Append training text to log file."""
    with open(log_file, "a") as f:
        f.write(text + "\n")


def save_checkpoint(model, optimizer, epoch, val_acc, best_model_path, best_val_acc):
    """Save model each epoch and best model."""
    ckpt_path = f"outputs/checkpoints/epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": val_acc,
    }, ckpt_path)
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), best_model_path)
        best_val_acc = val_acc
    return best_val_acc
