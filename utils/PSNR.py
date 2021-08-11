from PIL import Image
import numpy as np
import math
import os

def psnr(im1, im2):
    im1 = np.asarray(im1, dtype=np.float64)
    im2 = np.asarray(im2, dtype=np.float64)
    mse = np.mean(np.square(im1 - im2))
    return 10. * np.log10(np.square(255.) / mse)
