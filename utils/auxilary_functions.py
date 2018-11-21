import numpy as np
from skimage.transform import resize
from skimage import io as img_io
from skimage.color import rgb2gray
import torch
import torch.nn.functional as F


def grd_mgn(img):

    sobel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().to(img.device)
    gy = F.conv2d(img, sobel.view(1, 1, 3, 3), padding=1)
    gx = F.conv2d(img, sobel.t().view(1, 1, 3, 3), padding=1)
    mgn = torch.sqrt(gx ** 2 + gy ** 2)

    return mgn

def image_resize(img, height=None, width=None):

    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale*img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale*img.shape[0])

    img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    return img


def centered(word_img, tsize):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = diff_h/2
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = diff_w / 2
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    mv = np.median(word_img)
    word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=mv)
    return word_img