from skimage.io import imread, imsave
import numpy as np
from skimage.color import rgb2gray, rgb2yuv, yuv2rgb
from skimage import exposure, img_as_float, img_as_ubyte
from scipy.ndimage.filters import gaussian_filter
import os
import sys
import bisect
import argparse

def stretch(image, down = 0.0, up = 1.0):
    img = image.copy()
    img -= down
    img /= (up - down)
    return np.clip(img, 0, 1)

def get_down_up(img, thrs=(1, 12)):
    cum_hist = exposure.histogram(img)[0].cumsum()
    n_pixels = cum_hist[-1]
    cum_hist = cum_hist / n_pixels
    down = bisect.bisect_left(cum_hist, thrs[0] / 100.0)
    up = bisect.bisect_right(cum_hist, thrs[1] / 100.0)
    return down, up

def auto_white_balance_chnl(img, thrs=(1, 12)):
    down, up = get_down_up(img, thrs=thrs)
    down /= 255.0
    up   /= 255.0
    result = stretch(img, down, up)
    return result

def auto_white_balance(img, thrs=(1, 12)):
    result = img.copy()
    if len(img.shape) == 2:
        result = auto_white_balance_chnl(result, thrs=thrs)
        return result
    for chnl in range(img.shape[2]):
        result[:, :, chnl] = auto_white_balance_chnl(result[:, :, chnl], thrs=thrs)
    return result

def normalize_lighting(img, sigma=40):
    blurred = gaussian_filter(img, sigma)
    result = np.log(img / blurred + 1e-7) + 0.5
    result -= result.min()
    result /= result.max()
    return np.clip(result, 0, 1)

def change_extension(filename, extension='png'):
    return filename.split('.')[-2] + '.' + extension

def main():
    parser = argparse.ArgumentParser(description='Image enchancement.\
        Restoration of lighcting and contrast stretching.')
    parser.add_argument('path', nargs='+',
        help='Input filename')
    parser.add_argument('-g', '--gray', action='store_true',
        help='Make a grayscale image')
    parser.add_argument('-s', '--sigma', type=float,
        help='Value of sigma for gaussian filter')
    parser.add_argument('-o', nargs='+',
        help='Output filename')
    parser.add_argument('-t', '--thrs', nargs=2, type=float,
        default=(0.1, 99.9),
        help='The low and high thresholds for contrast stretching')
    parser.add_argument('-f', '--format',
        help='Output image extension')

    args = parser.parse_args()

    imgs = list(map(imread, args.path))
    imgs = list(map(img_as_float, imgs))
    if args.gray:
        imgs = list(map(rgb2gray, imgs))

    sigma = args.sigma
    if args.sigma is None:
        sigma = imgs[0].shape[0] / 30
    imgs = list(map(lambda img: normalize_lighting(img, sigma), imgs))
    imgs = list(map(lambda img: auto_white_balance(img, args.thrs), imgs))

    outpath = args.path
    if args.o is not None and len(args.o) == len(args.path):
        outpath = args.o
    if args.format is not None:
        outpath = list(map(lambda x: change_extension(x, args.format), outpath))
    list(map(lambda x: imsave(*x, quality=0.9), zip(outpath, imgs)))

if __name__ == "__main__":
    main()