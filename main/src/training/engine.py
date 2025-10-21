from torch.optim import Adam
from torch import nn
import torch
from tqdm.auto import tqdm
from torch_snippets.torch_loader import Report
from src.models import get_model
from src.data.loader import loader,PlantDataset,transforms
from src.data.Eda import set_up
from sklearn.metrics import accuracy_score

labels_dict = set_up()
trn_ds,val_ds = PlantDataset(main_path=main_folder_path,labels_dict=labels_dict,tf=transforms) # type: ignore
trn_dl,val_dl = loader(train_dataset=trn_ds,valid_dataset=val_ds)
model = get_model()
optimizer = Adam(lr=1e-3,params=model.parameters())
criterion = nn.CrossEntropyLoss()

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

n_epochs = 50
log =Report(n_epochs=n_epochs)
def train(trn_dl,val_dl):
    for epoch in tqdm(range(n_epochs)):
        _n = len(trn_dl)
        for ix,inputs  in enumerate(trn_dl):
            loss,acc = train_batch(inputs,model,optimizer,criterion)
            pos = (epoch+(ix+1)/_n)
            log.record(pos,trn_loss=loss,trn_acc = acc )

        _n = len(val_dl)   
        for ix,inputs  in enumerate(val_dl):
            loss,acc = validate_batch(model,inputs,criterion)
            pos = (epoch+(ix+1)/_n)
            log.record(pos,val_loss=loss,val_acc = acc)
    
