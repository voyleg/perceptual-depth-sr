from PIL import Image
import numpy as np


def modcrop(imgs, modulo):
    # input image shape (h, w, channels)
    tmpsz = imgs.shape
    sz = [tmpsz[0], tmpsz[1]]
    sz[0] -= sz[0] % modulo
    sz[1] -= sz[1] % modulo

    if len(tmpsz) == 3:
        imgs = imgs[:sz[0], :sz[1], :]
    else:
        imgs = imgs[:sz[0], :sz[1]]
    return imgs


def resample(im, scale):
    im_Dl = Image.fromarray(im)
    old_size = im_Dl.size
    new_size = int(old_size[0] * scale), int(old_size[1] * scale)
    im_Dl = im_Dl.resize(new_size, resample=Image.BICUBIC)
    return np.array(im_Dl)


def normalize(ar):
    vmin = ar[np.isfinite(ar)].min()
    vmax = ar[np.isfinite(ar)].max()
    assert vmin != vmax
    return (ar - vmin) / (vmax - vmin), vmin, vmax


def rgb2ycbcr(rgb):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = rgb.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return rgb
