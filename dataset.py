import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

def center(img):
    x = np.array(img)
    return (x * np.arange(img.size[0])).sum() / x.sum(), (x.T * np.arange(img.size[1])).sum() / x.sum()

W = 80
H = 80
font_size = 64
font_color = (255,)
margin = 10
unicode_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", font_size)
IMAGE_ROOT = '/home/wshi/git/misc/ocr/label/data/02111843.cn_tif'

def get_ideal_image(word):
    img1 = Image.new('L', (80, 80))
    d = ImageDraw.Draw(img1)
    w, h = unicode_font.getsize(word)
    d.text(((W-w)/2,(H-h)/2 - 14), word, font = unicode_font, fill = font_color)
    ex, ey = center(img1)
    E = min(W, H)
    img1 = ImageOps.expand(img1, border = E).crop((ex - E // 2 + E, ey - E // 2 + E, ex + E // 2 + E, ey + E //2 + E)).resize((W, H), Image.ANTIALIAS)
    return img1

def normalize_raw_image(img):
    ax, ay = center(img)
    E = img.size[1] + margin
    img3 = ImageOps.expand(img, border = E).crop((ax - E // 2 + E, ay - E //2 + E, ax + E // 2 + E, ay + E //2 + E)).resize((W, H), Image.ANTIALIAS)
    return img3
 

class WordDataset(data.Dataset):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.transform = transforms.ToTensor()

    def get_image(self, index):
        pid, left, upper, right, lower, word = self.dictionary[index]
        img = Image.open(os.path.join(IMAGE_ROOT, pid)).convert("L")
        return ImageOps.invert(img.crop((left, upper, right, lower)))


    def __getitem__(self, index):
        pid, left, upper, right, lower, word = self.dictionary[index]
        img1 = get_ideal_image(word)
        img2 = self.get_image(index)
        img2 = img2.rotate(2 * np.random.uniform(-1, 1))
        img2 = img2.resize((img2.size[0] + np.random.randint(-2, 2) * 3, img2.size[1] + np.random.randint(-2, 2)), Image.ANTIALIAS)
        img3 = normalize_raw_image(img2)
        #bleft = min(a for a in np.argmin(np.cumsum(1 - np.array(img2) / 255, 1) == 0, 1) if a > 0)
        #bright = np.array(img2).shape[1] - min(a for a in np.argmin(np.cumsum(1 - np.array(img2)[:, ::-1] / 255, 1) == 0, 1) if a > 0)
        #img3 = img2.crop((max(bleft - margin, 0), 0, min(bright + margin, img2.size[0]), img2.size[1])).resize((W, H), Image.ANTIALIAS)
        return self.transform(img3), self.transform(img1)

    def __len__(self):
        return len(self.dictionary)

small_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 24)
W1 = 32
H1 = 32

def change_contrast(img, factor, base = 128):
    def contrast(c):
        return base + factor * (c - base)
    return img.point(contrast)

class ClearDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        with open('word.txt') as f:
            self.dictionary = list(f.read().strip())
        transform_list = [
            transforms.RandomRotation(3, resample = Image.BICUBIC),
            #transforms.RandomResizedCrop(80, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
                ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        word = self.dictionary[index]
        img1 = Image.new('L', (80, 80))
        d = ImageDraw.Draw(img1)
        w, h = unicode_font.getsize(word)
        d.text(((W-w)/2,(H-h)/2 - 14), word, font = unicode_font, fill = font_color)

        #img2 = img1.resize((W1, H1), Image.ANTIALIAS).resize((W, H), Image.ANTIALIAS)
        img2 = Image.new('L', (W1, H1))
        d = ImageDraw.Draw(img2)
        w, h = small_font.getsize(word)
        d.text(((W1-w)/2, (H1-h)/2 - 3), word, font = small_font, fill = font_color)
        try:
            bleft = min(a for a in np.argmin(np.cumsum(np.array(img2), 1) == 0, 1) if a > 0)
            bright = min(a for a in np.argmin(np.cumsum(np.array(img2)[:, ::-1], 1) == 0, 1) if a > 0)
            btop = min(a for a in np.argmin(np.cumsum(np.array(img2), 0) == 0, 0) if a > 0)
            bbotton = min(a for a in np.argmin(np.cumsum(np.array(img2)[::-1, :], 0) == 0, 0) if a > 0)
            pad = max(0, min(bleft, bright, btop, bbotton) - margin)
        except ValueError:
            pad = 0
        img3 = img2.crop((pad, pad, img2.size[0] - pad, img2.size[1] - pad)).resize((W, H), Image.ANTIALIAS)
        return self.transform(img3), transforms.ToTensor()(img1)

    def __len__(self):
        return len(self.dictionary)

with open('word.txt') as f:
    dictionary = [get_ideal_image(w) for w in f.read().strip()]

def random_word(batch_size): 
    words = [dictionary[i] for i in np.random.choice(len(dictionary), batch_size)] 
    return torch.cat([transforms.ToTensor()(w) for w in words], 0)[:, None]

if __name__ == '__main__':
    ds = WordDataset('word.txt')
    print(len(ds))
    print(ds[0])
