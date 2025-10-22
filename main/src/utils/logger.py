import os
import torch
from datetime import datetime
import os 

def create_logger(base_dir=None):
    """Create log and checkpoint folders + filenames."""
    if  base_dir:
        base_dir = os.getcwd()
    logs_dir = os.path.join(base_dir,'outputs','logs')    
    ck_dir = os.path.join(base_dir,'outputs','checkpoints')    
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ck_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file =  os.path.join(logs_dir,f'train_{timestamp}.log')
    best_model_path = os.path.join(ck_dir,f"best_model.pt")
    return log_file, best_model_path,base_dir


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
