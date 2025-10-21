###load config file
import yaml

def load_config(path='./configs/train.yaml'):
    with open(path,'r') as file:
        config = yaml.safe_load(f)
    return config