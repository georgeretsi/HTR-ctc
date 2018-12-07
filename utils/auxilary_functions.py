import numpy as np
from skimage.transform import resize
from skimage import io as img_io
from skimage.color import rgb2gray
import torch
import torch.nn.functional as F


# morphological operations suppurted by cuda
def torch_morphological(img, kernel_size, mode='dilation'):

    # img : batches x 1 x h x w
    if mode == 'dilation':
        img = F.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=kernel_size/2)
    if mode == 'erosion':
        img = -F.max_pool2d(-img, kernel_size=kernel_size, stride=1, padding=kernel_size/2)
    if mode == 'closing':
        img = F.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=kernel_size/2)
        img = -F.max_pool2d(-img, kernel_size=kernel_size, stride=1, padding=kernel_size/2)
    if mode == 'opening':
        img = -F.max_pool2d(-img, kernel_size=kernel_size, stride=1, padding=kernel_size/2)
        img = F.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=kernel_size/2)

    return img


def affine(img):

    h, w = img.size(2), img.size(3)
    g = torch.stack([
        torch.linspace(-w/2, w/2, w).view(1, -1).repeat(h, 1),
        torch.linspace(-h/2, h/2, h).view(-1, 1).repeat(1, w),
    ], 2)

    scale = 1 #np.random.uniform(.9, 1.1)
    x_prop = 1 #np.random.uniform(.9, 1.1)
    rotate = np.deg2rad(2 * np.random.randn())
    slant = np.random.randn()

    # ng = ng + .05 * torch.randn(ng.size())
    ng = scale * g
    ng[:, :, 0] = x_prop * ng[:, :, 0]
    ng[:, :, 0] = ng[:, :, 0] + slant * 40 * ng[:, :, 1] / h

    R = torch.from_numpy(
        np.asarray([np.cos(rotate), -np.sin(rotate), np.sin(rotate), np.cos(rotate)])).float().view(2, 2)
    ng = torch.mm(ng.view(-1, 2), R).view_as(g)
    ng[:, :, 0] = 2 * ng[:, :, 0] / w
    ng[:, :, 1] = 2 * ng[:, :, 1] / h

    nimg = F.grid_sample(img, ng.unsqueeze(0).to(img.device), padding_mode='border')

    return nimg


def torch_augm(img):

    # kernel radius
    r = np.random.randint(0, 2)
    if r > 0:
        mode = np.random.choice(['dilation', 'erosion', 'opening', 'closing'])
        img = torch_morphological(img, 2*r+1, mode)

    img = affine(img)

    return img.detach()


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