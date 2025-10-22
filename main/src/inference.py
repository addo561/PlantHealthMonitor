#######################
    #inference
#####################
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_path = '/Users/user/Downloads/PlantHealthMonitor/data/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG'


image = Image.open(image_path)
image = np.array(image)
plt.imshow(image)
plt.show()


