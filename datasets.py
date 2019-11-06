import glob
import random
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

import torchvision.transforms as transforms
class ConsistentRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = None

    def __call__(self, img):
        if self.cache is not None:
            i, j, h, w = self.cache
        else:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            self.cache = i, j, h, w
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __enter__(self):
        self.cache = None
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.cache = None



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
    def __init__(self, root, unaligned=False, mode='train', limit = MAX):
        self.random_transform = ConsistentRandomResizedCrop(size = (W, H), scale = (0.8, 1.2), ratio = (0.8, 1.2))
        transforms_ = [
                transforms.Pad(10),
                self.random_transform,
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ]

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
        self.weights = self.weights / self.weights.mean()
        self.reset()

    def reset(self):
        self.subst = self.mapping.set_index('word').sample(n = self.limit, weights=self.weights).reset_index()

    def __getitem__(self, index):
        row = self.subst.iloc[index]
        with self.random_transform:
            item_A = self.transform(get_image(os.path.join(self.root, self.mode, 'A', '%s.tif' % row['src'])))
            item_B = self.transform(get_image(os.path.join(self.root, self.mode, 'B', '%s.tif' % row['tgt'])))

        return (item_A, item_B, row.ordinal)
        #return {'A': item_A, 'B': item_B, 'index': index}

    def __len__(self):
        return len(self.subst)

    def update_weights(self, scores):
        scores = scores.sort_values()
        pidx = self.mapping.loc[scores.head(len(scores)//4).index].word
        self.weights.loc[pidx] = np.maximum(self.weights.loc[pidx] * 0.999, 0.25)
        pidx = self.mapping.loc[scores.tail(len(scores)//4).index].word
        self.weights.loc[pidx] = np.minimum(self.weights.loc[pidx] * 1.001, 4)



