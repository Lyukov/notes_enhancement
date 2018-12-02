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

def get_down_up(img, thrs=(0.7, 1.0)):
    hist     = exposure.histogram(img)[0]
    histmax = np.argmax(hist)
    down = histmax * thrs[0]
    up   = histmax * thrs[1]
    return down, up

def auto_white_balance_chnl(img, thrs=(0.7, 1.0)):
    h, w = img.shape
    down, up = get_down_up(img, thrs=thrs)
    down /= 255.0
    up   /= 255.0
    result = stretch(img, down, up)
    return result

def auto_white_balance(img, thrs=(0.7, 1.0)):
    result = img.copy()
    if len(img.shape) == 2:
        result = auto_white_balance_chnl(result, thrs=thrs)
        return result
    for chnl in range(img.shape[2]):
        result[:, :, chnl] = auto_white_balance_chnl(result[:, :, chnl], thrs=thrs)
    return result

def normalize_lighting(img, sigma=40):
    blurred = gaussian_filter(img, sigma)
    result = img / (blurred + 1e-7)
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
        default=(0.7, 1.0),
        help='The low and high thresholds for contrast stretching')
    parser.add_argument('-f', '--format',
        help='Output image extension')

    args = parser.parse_args()

    outpaths = args.path
    if args.o is not None and len(args.o) == len(args.path):
        outpaths = args.o
    if args.format is not None:
        outpaths = list(map(lambda x: change_extension(x, args.format), outpaths))
    sigma = args.sigma
    if args.sigma is None:
        sigma = imread(args.path[0]).shape[0] / 30

    for img_path, out_path in zip(args.path, outpaths):
        img = img_as_float(imread(img_path))
        if args.gray:
            img = rgb2gray(img)
        #img = normalize_lighting(img, sigma)
        img = auto_white_balance(img, args.thrs)
        imsave(out_path, img)
        print(img_path + ": done")

if __name__ == "__main__":
    main()