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

labels_dict = set_up()
trn_ds,val_ds = PlantDataset(main_path=main_folder_path,labels_dict=labels_dict,tf=transforms) # type: ignore
trn_dl,val_dl = loader(train_dataset=trn_ds,valid_dataset=val_ds)
model = get_model()
optimizer = Adam(lr=1e-3,params=model.parameters())
criterion = nn.CrossEntropyLoss()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"outputs/logs/train_{timestamp}.log"
best_model_path = "outputs/checkpoints/best_model.pt"




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
    n_epochs = 50
    log =Report(n_epochs=n_epochs)
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

        with open(log_file,'a') as f:
            f.write(
                f'Epoch {epoch+1}/{n_epochs} |'
                f'Train_loss :{trn_loss}  Train_accuracy : {trn_acc}|' 
                f'val_loss :{val_loss}  val_accuracy : {val_acc}|' 
            )
        ck_path = f"outputs/checkpoints/Epoch_{epoch+1}.pt"
        torch.save({
            'epoch':  epoch + 1,
            'model_state': model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'val_acc':val_acc
        },ck_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
    print('Training completed') 
    log.plot()
    log.save(f"outputs/logs/report_{timestamp}.png")
