import cv2
import numpy as np
from scipy.fftpack import dct

# list input image
pictures = [
    'output.bmp',
    'gaussianwn.bmp',
    'gaussianlp.bmp',
    'cutted.bmp',
    'rotated.bmp',
    'compressed.jpg'
]

# Original and secret image sizes
original_row = 480
original_col = 640

secret_row = 160
secret_col = 160

# 2 way DCT 
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

for filename in pictures:
    # read grayscale
    marked = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # convert to float for DCT
    marked_dcted = dct2(np.float32(marked))

    #Initialize secret image extraction
    secret = np.zeros((secret_row, secret_col), dtype=np.uint8)

    # Browse every secret pixel
    for i in range(secret_row):
        for j in range(secret_col):
            x1 = original_row - 2 * i - 1
            x2 = original_row - 2 * i - 2
            y = original_col - 2 * j - 1

            if marked_dcted[x2, y] < marked_dcted[x1, y]:
                secret[i, j] = 255  # white pixel
            else:
                secret[i, j] = 0    # black pixel

    # output image
    out_name = f'secret_{filename}.bmp'
    cv2.imwrite(out_name, secret)
