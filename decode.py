import cv2
import numpy as np
from scipy.fftpack import dct

# Danh sách tên file ảnh đầu vào
pictures = [
    'output.bmp',
    'gaussianwn.bmp',
    'gaussianlp.bmp',
    'cutted.bmp',
    'rotated.bmp',
    'compressed.jpg'
]

# Kích thước ảnh gốc và ảnh bí mật
original_row = 480
original_col = 640

secret_row = 160
secret_col = 160

# Hàm DCT 2 chiều
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# Lặp qua từng ảnh
for filename in pictures:
    # Đọc ảnh grayscale
    marked = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Chuyển về float để DCT
    marked_dcted = dct2(np.float32(marked))

    # Khởi tạo ảnh bí mật trích xuất
    secret = np.zeros((secret_row, secret_col), dtype=np.uint8)

    # Duyệt từng pixel bí mật
    for i in range(secret_row):
        for j in range(secret_col):
            x1 = original_row - 2 * i - 1
            x2 = original_row - 2 * i - 2
            y = original_col - 2 * j - 1

            if marked_dcted[x2, y] < marked_dcted[x1, y]:
                secret[i, j] = 255  # pixel trắng
            else:
                secret[i, j] = 0    # pixel đen

    # Tạo tên ảnh đầu ra
    out_name = f'secret_{filename}.bmp'
    cv2.imwrite(out_name, secret)
