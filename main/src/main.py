"""
main  file
"""
import torch
import importlib
import main.src.data.Eda as Eda
importlib.reload(Eda)
from main.src.data.Eda import set_up,plot_image_sizes

labels_dict = set_up(main_folder_path,show=True) # type: ignore
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device