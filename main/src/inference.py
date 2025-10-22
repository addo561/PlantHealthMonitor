#######################
    #inference
#####################
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms  import transforms
from torchvision.models import vit_b_16,ViT_B_16_Weights

labels = {'Tomato_Septoria_leaf_spot': 0,
 'Tomato__Tomato_YellowLeaf__Curl_Virus': 1,
 'Tomato__Target_Spot': 2,
 'Tomato__Tomato_mosaic_virus': 3,
 'Pepper__bell___Bacterial_spot': 4,
 'Tomato_Late_blight': 5,
 'Pepper__bell___healthy': 6,
 'Tomato_Leaf_Mold': 7,
 'Tomato_Bacterial_spot': 8,
 'Tomato_Early_blight': 9,
 'Potato___Early_blight': 10,
 'Potato___healthy': 11,
 'Tomato_healthy': 12,
 'Tomato_Spider_mites_Two_spotted_spider_mite': 13,
 'Potato___Late_blight': 14}

labels = {v:k for k,v in labels.items()}
image_path = '/Users/user/Downloads/PlantHealthMonitor/data/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG'
tf =  transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
   transforms.Resize((224,224)),
])

classes =  15
model = vit_b_16(weights = ViT_B_16_Weights)
model.heads = nn.Sequential(
    nn.Linear(in_features=768,out_features=classes)
    )
for param in model.parameters():
    param.requires_grad  = False
ck = '/Users/user/Downloads/PlantHealthMonitor/main/outputs/checkpoints/best_model.pt'
model.load_state_dict(torch.load(ck,map_location='cpu'))

image = Image.open(image_path)
image = np.array(image)
image_tf = tf(image)

with torch.inference_mode():
    model.eval()
    pred = model(image_tf[None,:,:,:])
pred = torch.argmax(pred,dim=1).item()
print(f'class {pred}')    



plt.imshow(image)
plt.title(f'{labels[pred]} {image.shape}')
plt.axis('off')
plt.show()


