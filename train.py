import os
import torch

import torch.utils.data

import torchvision.transforms as transforms

from loader import ImageFolder
from PIL import Image

CUDA = True

DATA_ROOT = "./dataset/"
OUT_PATH = "./outputs"

IMAGE_SIZE = 256
BATCH_SIZE = 1

dataset = ImageFolder(root=DATA_ROOT, transform=transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE*1.12), Image.BICUBIC),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

try:
    os.makedirs(os.path.join(OUT_PATH,"A"))
    os.makedirs(os.path.join(OUT_PATH,"B"))
except OSError:
    pass

try: 
    os.os.makedirs(os.path.join("weights"))
except OSError:
    pass

device = torch.device("cuda:0" if CUDA else "cpu")
