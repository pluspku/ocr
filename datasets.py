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

W=H=128

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
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', limit = MAX):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.limit = limit
        self.root = root
        self.mode = mode

        if os.path.exists(os.path.join(root, mode, "mapping.csv")):
            self.mapping = pd.read_csv(os.path.join(root, mode, "mapping.csv"))
            self.mapping['ordinal'] = self.mapping.index
        else:
            raise Exception("%s does not exists" % os.path.join(root, mode, "mapping.csv"))
        self.weights = np.sqrt(self.mapping.groupby("word").size())
        self.reset()

    def reset(self):
        self.subst = self.mapping.set_index('word').sample(n = self.limit, weights=self.weights).reset_index()

    def __getitem__(self, index):
        row = self.subst.iloc[index]
        item_A = self.transform(get_image(os.path.join(self.root, self.mode, 'A', '%s.tif' % row['src'])))
        item_B = self.transform(get_image(os.path.join(self.root, self.mode, 'B', '%s.tif' % row['tgt'])))

        return (item_A, item_B, row.ordinal)
        #return {'A': item_A, 'B': item_B, 'index': index}

    def __len__(self):
        return len(self.subst)

    def update_weights(self, scores):
        scores = scores.sort_values()
        self.weights.loc[self.mapping.loc[scores.head(len(scores)//4).index].word] *= 0.99
        self.weights.loc[self.mapping.loc[scores.tail(len(scores)//4).index].word] *= 1.01



