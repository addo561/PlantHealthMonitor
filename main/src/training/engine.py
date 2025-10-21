from torch.optim import Adam
from torch import nn
import torch
from tqdm.auto import tqdm
from torch_snippets.torch_loader import Report
from src.models import get_model
from src.data.loader import loader,PlantDataset,transforms
from src.data.Eda import set_up
from sklearn.metrics import accuracy_score
import datetime
from  src.utils.config  import load_config
from  src.utils.logger import log_to_file,create_logger,save_checkpoint

labels_dict = set_up()
trn_dl,val_dl = loader()
model = get_model()
cfg  =  load_config()
lr = cfg['train']['learning_rate']
optimizer = Adam(lr=lr,params=model.parameters())
criterion = nn.CrossEntropyLoss()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")





def train_batch(inputs,model,optimizer,criterion):
    model.train()
    n = len(trn_dl)
    im,l =  inputs
    pred = model(im)
    loss = criterion(pred,l)
    accurary = accuracy_score(l,pred.argmax(dim=1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(),accurary.item()

@torch.no_grad()
def validate_batch(model,inputs,criterion):
    model.eval()
    im,l = inputs
    pred = model(im)
    loss = criterion(pred,l)
    accurary = accuracy_score(l,pred.argmax(dim=1))
    return loss.item() ,accurary.item()


def train(trn_dl,val_dl):
    n_epochs = cfg['train']['epochs']
    log =Report(n_epochs=n_epochs)
    best_val_acc = 0
    log_file, best_model_path = create_logger()
    for epoch in tqdm(range(n_epochs)):
        trn_loss,trn_acc,val_loss,val_acc =  0,0,0,0
        _n = len(trn_dl)
        for inputs  in trn_dl:
            loss,acc = train_batch(inputs,model,optimizer,criterion)
            trn_loss += loss
            trn_acc += acc
        trn_loss /= _n
        trn_acc /= _n

        _n = len(val_dl)   
        for inputs  in val_dl:
            loss,acc = validate_batch(model,inputs,criterion)
            val_loss += loss
            val_acc += acc
        val_loss /=  _n
        val_acc /= _n

        log.record(epoch+1,trn_loss=loss,trn_acc = acc,val_loss=val_loss,val_acc=val_acc )
        log.report_avgs(epoch+1)

        text = f"Epoch {epoch+1}: Train Acc={trn_acc:.4f}, Val Acc={val_acc:.4f}"
        log_to_file(log_file,text=text)

        save_checkpoint(model,optimizer,epoch,val_acc,best_model_path,best_val_acc)
    print('Training completed') 
    log.plot()
    log.save(f"outputs/logs/report_{timestamp}.png")
