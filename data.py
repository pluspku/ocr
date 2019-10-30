from torchvision import transforms
from datasets import ImageDataset, W, H
import os

transforms_ = [
        transforms.Pad(10),
        transforms.RandomResizedCrop((W, H), scale = (0.9, 1.1), ratio = (0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ]

root = "datasets/rlws"

def get_training_set():
    return ImageDataset(root, transforms_, unaligned = False, mode = 'train', limit = 6000)

def get_test_set():
    return ImageDataset(root, transforms_, unaligned = False, mode = 'test', limit = 500)

