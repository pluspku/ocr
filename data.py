from datasets import ImageDataset
import os

root = "datasets/hlm2"

def get_training_set():
    return ImageDataset(root, unaligned = False, mode = 'train', limit = 6000)

def get_test_set():
    return ImageDataset(root, unaligned = False, mode = 'test', limit = 500)

