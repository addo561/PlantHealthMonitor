from torch.optim import Adam
from torch_snippets.torch_loader import Report
from src.models import get_model
from src.data.loader import loader,PlantDataset,transforms
from src.data.Eda import set_up

labels_dict = set_up()
trn_ds,val_ds = PlantDataset(main_path=main_folder_path,labels_dict=labels_dict,tf=transforms) # type: ignore
trn_dl,val_dl = loader(train_dataset=trn_ds,valid_dataset=val_ds)
model = get_model()
optimizer = Adam(lr=1e-3,params=model.parameters())

def train_batch(inputs,model,optimizer,loss_fn):
    n = len(trn_dl)
