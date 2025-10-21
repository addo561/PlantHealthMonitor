from torchinfo import summary
from torch import nn
from torchvision.models import ViT_B_16_Weights,vit_b_16

def get_model():
  weights = ViT_B_16_Weights.IMAGENET1K_V1#init weights
  model =  vit_b_16(weights=weights) #main model
  for param in  model.parameters(): #freeze base layers
    param.requires_grad = False
  model.heads = nn.Sequential(
      nn.Linear(
          in_features=768,
          out_features=14)
  )

  return  model

#checking model architecture  
model = get_model()
summary(
      model=model,
      input_size = (32,3,224,224),
      col_names=["input_size", "output_size", "num_params", "trainable"],
  )
