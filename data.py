from torchvision import transforms
from datasets import ImageDataset
import os

transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ]

root = "datasets/hlm2"

def get_training_set():
    return ImageDataset(root, transforms_, unaligned = False, mode = 'train', limit = 6000)

def get_test_set():
    return ImageDataset(root, transforms_, unaligned = False, mode = 'test', limit = 500)

