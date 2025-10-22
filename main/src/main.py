"""
main  file
"""
import torch
from torch import nn
import importlib
import main.src.data.Eda as Eda
importlib.reload(Eda)
from main.src.data.Eda import set_up,plot_image_sizes
from main.src.data.loader import loader,PlantDataset
from sklearn.model_selection import train_test_split
from torchvision.transforms  import transforms
from main.src.models.model import get_model
from torchinfo import summary

labels_dict = set_up(main_folder_path,show=True) # type: ignore
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# Define transformations
tf =  transforms.Compose([
   transforms.ToTensor(),
   #transforms.CenterCrop(10) ,
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
   transforms.Resize((224,224)),
   transforms.ColorJitter(brightness=0.3)
])

# Instantiate the dataset 
dataset = PlantDataset(main_path=main_folder_path,labels_dict=labels_dict,tf=tf) # type: ignore

# Split the dataset
train_size =  int(len(dataset) *  0.7)
valid_size =  len(dataset) -  train_size
train_dataset,valid_dataset = torch.utils.data.random_split(dataset,[train_size,valid_size])


# Create the dataloaders
trn_dl,val_dl  = loader(train_dataset,valid_dataset)


summary(
      model=get_model(),
      input_size = (32,3,224,224),
      col_names=["input_size", "output_size", "num_params", "trainable"],
  )

model = get_model().to(device)

from main.src.utils.config import load_config
from main.src.training.engine import train
from main.src.utils.logger  import create_logger,log_to_file,save_checkpoint
from torch.optim import Adam

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load cfg
cfg = load_config('/content/PlantHealthMonitor/main/configs/train.yaml')

# prepare data
dataset = PlantDataset(main_path=main_folder_path, labels_dict=labels_dict, tf=tf) # type: ignore
train_size = int(len(dataset) * 0.7)
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
trn_dl, val_dl = loader(train_dataset, valid_dataset)

# model + optimizer + criterion
model = get_model().to(device)
optimizer = Adam(model.parameters(), lr=cfg['train']['learning_rate'])
criterion = nn.CrossEntropyLoss().to(device)

# training
best_val_acc = train(trn_dl, val_dl, cfg, device, optimizer, criterion, model)
print("Best val acc:", best_val_acc)

