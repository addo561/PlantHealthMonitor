from pathlib import Path
import torch
import matplotlib.pyplot as plt

def add_path(path_from_parent: str):
    """
    Add a subdirectory inside the project to Python's sys.path for dynamic imports.

    Args:
        path_from_parent (str): Relative path from the project root (e.g., 'src' or 'src/utils')
    """
    project_root = Path("/content/PlantHealthMonitor")
    target_path = project_root / path_from_parent
    target_path.touch(exist_ok=True)
    print('Done')

    
def show_samples(dataset,labels_dict):
  randoms = torch.randint(0,20000,size=(10,)) #get random  indices
  fig,ax = plt.subplots(2,5,figsize=(15,7)) #plot
  ax = ax.flatten()
  for i,idx in enumerate(randoms):
    image,label  = dataset[idx]  #get label and image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    unnormalized_image = image*torch.tensor(std).view(3,1,1) + torch.tensor(std).view(3,1,1)
    unnormalized_image.shape,label

    label  =  label.item() #get label as int
    ax[i].imshow(unnormalized_image.permute(1,2,0).numpy())
    for n,l  in labels_dict.items():
      if l==label:
        label = n #set plt title
    ax[i].set_title(f'{label.replace('__','-').replace('_','-')}')
    ax[i].axis('off')
  plt.subplots_adjust(wspace=0.5)
  plt.show()
