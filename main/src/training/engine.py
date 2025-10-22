from torch.optim import Adam
from torch import nn
import torch
from tqdm.auto import tqdm
from torch_snippets.torch_loader import Report
from sklearn.metrics import accuracy_score
import datetime
from src.utils.logger import log_to_file, create_logger, save_checkpoint

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



def train_batch(inputs, model, optimizer, criterion, device):
    model.train()
    im, l = inputs
    im, l = im.to(device), l.to(device)
    pred = model(im)
    loss = criterion(pred, l)
    accuracy = accuracy_score(l.cpu(), pred.argmax(dim=1).cpu())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), float(accuracy)


@torch.no_grad()
def validate_batch(model, inputs, criterion, device):
    model.eval()
    im, l = inputs
    im, l = im.to(device), l.to(device)
    pred = model(im)
    loss = criterion(pred, l)
    accuracy = accuracy_score(l.cpu(), pred.argmax(dim=1).cpu())
    return loss.item(), float(accuracy)


def train(trn_dl, val_dl, cfg, device, optimizer, criterion, model):
    n_epochs = cfg['train']['epochs']
    log = Report(n_epochs=n_epochs)
    best_val_acc = 0
    
    # FIX 1: Unpack all 3 return values
    log_file, best_model_path, _ = create_logger(base_dir=cfg['paths']['base_dir'])
    
    for epoch in tqdm(range(n_epochs)):
        trn_loss, trn_acc, val_loss, val_acc = 0, 0, 0, 0
        
        # Training loop
        _n = len(trn_dl)
        for inputs in trn_dl:
            loss, acc = train_batch(inputs, model, optimizer, criterion, device)
            trn_loss += loss
            trn_acc += acc
        trn_loss /= _n
        trn_acc /= _n
        
        # Validation loop
        _n = len(val_dl)
        for inputs in val_dl:
            loss, acc = validate_batch(model, inputs, criterion, device)
            val_loss += loss
            val_acc += acc
        val_loss /= _n
        val_acc /= _n
        
        log.record(epoch+1, trn_loss=trn_loss, trn_acc=trn_acc, 
                   val_loss=val_loss, val_acc=val_acc)
        log.report_avgs(epoch+1)
        
        # Logging
        text = f"Epoch {epoch+1}: Train Acc={trn_acc:.4f}, Val Acc={val_acc:.4f}"
        log_to_file(log_file, text=text)
        
        #  Update best_val_acc from save_checkpoint
        best_val_acc = save_checkpoint(model, optimizer, epoch, val_acc, 
                                        best_model_path, best_val_acc)
    
    print('Training completed')
    log.plot()
    log.save(f"outputs/logs/report_{timestamp}.png")
    
    return best_val_acc