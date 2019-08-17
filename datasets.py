import glob
import random
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

import torchvision.transforms as transforms

MAX = 6000
#MAX = 999999

W=H=80

def center(img):
    x = np.array(img)
    return (x * np.arange(img.size[0])).sum() / x.sum(), (x.T * np.arange(img.size[1])).sum() / x.sum()

margin = 10
def get_image(path):
    img = Image.open(path)
    ax, ay = center(img)
    E = img.size[1] + margin
    img3 = ImageOps.expand(img, border = E).crop((ax - E // 2 + E, ay - E //2 + E, ax + E // 2 + E, ay + E //2 + E)).resize((W, H), Image.ANTIALIAS)
    return img3



class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        if os.path.exists(os.path.join(root, mode, "mapping.csv")):
            self.mapping = pd.read_csv(os.path.join(root, mode, "mapping.csv"))
            self.files_A = self.mapping.src.apply(lambda x: os.path.join(root, mode, 'A', '%s.tif' % x)).tolist()
            self.files_B = self.mapping.tgt.apply(lambda x: os.path.join(root, mode, 'B', '%s.tif' % x)).tolist()
        else:
            raise Exception("%s does not exists" % os.path.join(root, mode, "mapping.csv"))

    def __getitem__(self, index):

        if self.unaligned:
            item_A = self.transform(get_image(self.files_A[random.randint(0, len(self.files_A) - 1)]))
            item_B = self.transform(get_image(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_A = self.transform(get_image(self.files_A[index % len(self.files_A)]))            
            item_B = self.transform(get_image(self.files_B[index % len(self.files_B)]))

        return (item_A, item_B)
        #return {'A': item_A, 'B': item_B, 'index': index}

    def __len__(self):
        return min(MAX, max(len(self.files_A), len(self.files_B)))


