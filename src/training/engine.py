from torch.optim import Adam
from torch_snippets.torch_loader import Report
from src.models import get_model
from src.data.loader import 


model = get_model()
optimizer = Adam(lr=1e-3,params=model.parameters())
def train_batch(inputs,model,optimizer,loss_fn):
