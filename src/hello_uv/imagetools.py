import numpy as np
from scipy import ndimage


def invert(img):
    return 255 - img

def adjust_contrast(img,factor):
    flat_img = img.copy()
    flat_img = flat_img.astype(float)
    img_mean = flat_img.mean()
    new = (flat_img - img_mean) * factor + img_mean
    new_clip = np.clip(new, 0, 255)
    clip_round = np.round(new_clip)
    return new_clip.astype(np.uint8)

def shift(img,dx,dy):
    shift_im = np.roll(img, (dy,dx), axis=(0,1))
    if dx > 0:
        shift_im[:,:dx] = 0
    elif dx < 0:
         shift_im[:, dx:] = 0
    if dy > 0:
        shift_im[:dy, :] = 0
    elif dx < 0:
        shift_im[dy:, :] = 0
    return shift_im

def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False, order=1)

    
    