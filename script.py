from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

W = 80
H = 80

W1 = 16
H1 = 16

font_size = 64
font_color = (255,)
unicode_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK.ttc", font_size)

def change_contrast(img, factor, base = 128):
    #factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return base + factor * (c - base)
    return img.point(contrast)


for word in tqdm(open('word.txt').read()):
    img1 = Image.new('L', (80, 80))
    d = ImageDraw.Draw(img1)
    w, h = unicode_font.getsize(word)
    d.text(((W-w)/2,(H-h)/2 - 14), word, font = unicode_font, fill = font_color)

    img2 = img1.resize((W1, H1), Image.ANTIALIAS).resize((W, H), Image.ANTIALIAS)
    img2 = change_contrast(img2, 2, 60)

