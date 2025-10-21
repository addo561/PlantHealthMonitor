import os
from tqdm.auto import tqdm
import dask.bag as bag
import  dask.diagnostics  as diagnostics
from PIL import Image,UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_up(
    main_folder_path:str,
    show: bool=False
):

  """Data understanding mainly
  Args:
      main_folder_path: parent folder  with images
      show: display class numbers (show class imbalance)
  Returns:
        Dict of  labels with number  of images in each plant  and name
  """
  #get number  of plant
  print(f"{len(os.listdir(main_folder_path))} plants in main folder")

  #number of  each plant with disease and  name
  if os.path.isdir(main_folder_path):
    labels = {} #labels
    label = 0
    quantities  = {} #image number
    for item_name in os.listdir(main_folder_path):
      print(f"{item_name} with {len(os.listdir(os.path.join(main_folder_path,item_name)))} images")
      #labels
      labels[item_name] = label #store image name  and label
      label+=1
      quantities[item_name] = len(os.listdir(os.path.join(main_folder_path,item_name))) #store number of images
  else:
    print('[INFO] use correct directory')
  sns.set()
  # check  image class imbalance
  plt.figure(figsize=(15,10))
  plt.bar(quantities.keys(),quantities.values())
  plt.title('check class Imabalance')
  plt.xlabel('Diseases')
  plt.ylabel('Number of Images')
  plt.xticks(rotation=90)
  plt.show()
  # get two sample images from each class
  if show:
    plt.figure(figsize=(20,20))
    plot_num  = 1
    for folder in  tqdm(os.listdir(main_folder_path)):
      image_name = os.listdir(os.path.join(main_folder_path,folder))
      for i in range(min(2,len(image_name))):
          image = Image.open(os.path.join(main_folder_path,folder,image_name[i]))  #image paths
          imagearray = np.array(image)
          plt.subplot(5,6,plot_num) #30 samples 2  for each
          plt.imshow(imagearray)
          plt.title(f'{folder.replace('__','_').replace('__','_')} \n Shape: {imagearray.shape}')
          plt.axis('off')
          plot_num+=1
    plt.subplots_adjust(wspace=0.9)
    plt.show()
  return labels

def get_dim(file):
  """get image dimensions
  Args:
      file: Image file
  Returns:
        height  and width of image
  """
  try:
    image = Image.open(file)#load file
    image = np.array(image)#change  to  numoy array
    h,w,c = image.shape # get height and  width of image
    return  h,w
  except (UnidentifiedImageError, OSError) as e:
    print(f'Not found,  skipping file {os.path.basename(file)}')
    return None,None


  #dictionary of label and directories
def plot_image_sizes(main_folder_path):
  """
Analyze image dimensions in given directories and visualize their size distribution.

For each directory, computes (height, width) of all images in parallel using Dask,
groups by unique dimensions, and plots a scatter of image sizes.

Args:
        main_folder_path: parent folder  with images
    Returns:
          scatter plots of image  sizes
"""
# get all folder of each label
  directories = {d:os.path.join(main_folder_path,d) for d in os.listdir(main_folder_path) }
  for c,d in directories.items():
    filepath = d
    filelist  = [os.path.join(filepath, file) for file in os.listdir(filepath)]
    dims =  bag.from_sequence(filelist).map(get_dim) #parallel  computing  (put all in list  bag to compute getdims in parallel)
    with diagnostics.ProgressBar():
      dims = dims.compute()#computing dims
    df = pd.DataFrame(dims,columns=['height','width'])
    sizes =  df.groupby(['height','width']).size().reset_index().rename(columns={0:'Count'}) #get new couunt columns of image with identical sizes
    sizes.plot.scatter(x='width',y='height')
    plt.title(f'{c.replace('__','-').replace('_','-')}')
    plt.show()

