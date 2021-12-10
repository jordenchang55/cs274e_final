import os

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from loader import ImageFolder
from PIL import Image

from models.cycle_gan import CycleGAN

CUDA = True

DATA_ROOT = "./datasets/horse2zebra"
OUT_PATH = "./outputs"

IMAGE_SIZE = 256
BATCH_SIZE = 1

train_A_root = os.path.join(DATA_ROOT, 'trainA')
train_B_root = os.path.join(DATA_ROOT, 'trainB')

#train_A_dataset = ImageFolder(root=train_A_root, transform=transforms.Compose([
#    transforms.Resize(int(IMAGE_SIZE * 1.12), Image.BICUBIC),
#    transforms.RandomCrop(IMAGE_SIZE),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#]))

#train_B_dataset = ImageFolder(root=train_B_root, transform=transforms.Compose([
#    transforms.Resize(int(IMAGE_SIZE * 1.12), Image.BICUBIC),
#    transforms.RandomCrop(IMAGE_SIZE),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#]))

#train_A_loader = torch.utils.data.DataLoader(train_A_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
#train_B_loader = torch.utils.data.DataLoader(train_B_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

train_dataset = ImageFolder(root_A=train_A_root, root_B=train_B_root, transform=transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.12), Image.BICUBIC),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


try:
    os.makedirs(os.path.join(OUT_PATH, "A"))
    os.makedirs(os.path.join(OUT_PATH, "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights"))
except OSError:
    pass

device = torch.device("cuda:3" if CUDA else "cpu")

model = CycleGAN(dataloader, 0.0002, 50, device=device, output_folder=OUT_PATH)
model.train(300)

