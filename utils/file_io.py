import yaml
from easydict import EasyDict as edict

def yaml_load(fileName):
    fc = None
    with open(fileName, 'r') as f:
        fc = edict(yaml.load(f, Loader=yaml.FullLoader))

    return fc