import cv2
import numpy as np
from scipy.fftpack import dct, idct

# Hàm DCT và IDCT 2 chiều
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# Đọc ảnh gốc (grayscale)
original = cv2.imread('original.bmp', cv2.IMREAD_GRAYSCALE)
original_row, original_col = original.shape

# Đọc ảnh chứa thông tin bí mật và chuyển thành nhị phân
secret_img = cv2.imread('./SecretPics/secret160.bmp', cv2.IMREAD_GRAYSCALE)
_, secret_bw = cv2.threshold(secret_img, 127, 1, cv2.THRESH_BINARY)
secret_row, secret_col = secret_bw.shape

# Thực hiện DCT
img_dcted = dct2(np.float32(original))
alpha = 15

# Ẩn ảnh bằng DCT
for i in range(secret_row):
    for j in range(secret_col):
        x1 = original_row - 2 * i - 1
        x2 = original_row - 2 * i - 2
        y = original_col - 2 * j - 1

        if secret_bw[i, j] == 0:
            if img_dcted[x2, y] < img_dcted[x1, y]:
                img_dcted[x1, y], img_dcted[x2, y] = img_dcted[x2, y], img_dcted[x1, y]
        else:
            if img_dcted[x2, y] > img_dcted[x1, y]:
                img_dcted[x1, y], img_dcted[x2, y] = img_dcted[x2, y], img_dcted[x1, y]

        # Điều chỉnh độ chênh lệch giữa 2 hệ số
        if img_dcted[x2, y] < img_dcted[x1, y]:
            img_dcted[x2, y] -= alpha
            img_dcted[x1, y] += alpha
        else:
            img_dcted[x2, y] += alpha
            img_dcted[x1, y] -= alpha

# IDCT và lưu ảnh đã nhúng
output = idct2(img_dcted)
output = np.clip(output, 0, 255).astype(np.uint8)
cv2.imwrite('output.bmp', output)
