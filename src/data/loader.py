from torchvision.transforms  import transforms
from  pathlib import Path
import torch
from torch.utils.data import DataLoader,Dataset,random_split
import os
from PIL import Image,UnidentifiedImageError
import numpy as np
from tqdm.auto import tqdm
from src.data.Eda import set_up

labels_dict = set_up()

#perform data augmentations and other other transformations
tf =  transforms.Compose([
   transforms.ToTensor(),
   #transforms.CenterCrop(10) ,
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
   transforms.Resize((224,224)),
   transforms.ColorJitter(brightness=0.3)
])
class PlantDataset(Dataset):
  """ Image dataset
   Attributes:
              paths: Images
              labels_dict: dictionary of image label and number
              tf: transformations
  """
  def __init__(self,main_path,labels_dict,tf=None):
    """Initializes instance based on paths and label
      Args:
          paths and label_dict: Defines if instance exhibits this preference.
          tf :transformations
    """
    self.main_path  = main_path
    self.labels_dict = labels_dict
    self.tf = tf
    self.image_paths = [] # List to store all image paths
    self.image_labels = [] # List to store all image labels

    # Populate the lists of image paths and labels
    for folder in tqdm(os.listdir(self.main_path), desc="Loading image paths"):
        folder_path = os.path.join(self.main_path, folder)
        if os.path.isdir(folder_path): # Ensure it's a directory
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                try:
                    Image.open(image_path).verify() # Verify if it's a valid image file
                    self.image_paths.append(image_path)
                    # label for the image based on the folder name
                    label = Path(image_path).parent.name
                    self.image_labels.append(self.labels_dict[label])
                except (IOError, UnidentifiedImageError) as e:
                    print(f"Skipping file {image_path}: {e}")


  def __len__(self):
    return  len(self.image_paths)  # Return the total number of valid images

  def __getitem__(self,idx):
    image_path = self.image_paths[idx]
    label = self.image_labels[idx]

    # Open the image
    image = Image.open(image_path).convert('RGB') # Convert to RGB to ensure 3 channels

    # Apply transformations if any
    if self.tf:
      image =  self.tf(image)

    return image, torch.tensor(label)


dataset = PlantDataset(main_folder_path,labels_dict,tf)
train_size =  int(len(dataset) *  0.7)
valid_size =  len(dataset) -  train_size
train_dataset,valid_dataset = random_split(dataset,[train_size,valid_size])
def loader(train_dataset,valid_dataset):
  trn_dl = DataLoader(
      train_dataset,
      batch_size=64,
      shuffle=True) #creating dataset with 64 batch size
  v_dl = DataLoader(
      valid_dataset,
      batch_size=64#creating dataset with 64 batch size ,set shuffle to False
  )
  return trn_dl,v_dl
