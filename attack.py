import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, rotate

# Đọc ảnh
marked = cv2.imread('output.bmp', cv2.IMREAD_GRAYSCALE)

# 1. Thêm nhiễu Gaussian
mean = 0
var = 0.02
sigma = var ** 0.5
gaussian_noise = np.random.normal(mean, sigma, marked.shape)
gaussianwn = marked + gaussian_noise * 255
gaussianwn = np.clip(gaussianwn, 0, 255).astype(np.uint8)
cv2.imwrite('gaussianwn.bmp', gaussianwn)

# 2. Lọc Gaussian (low pass)
# Tương đương fspecial('gaussian', [4, 4], 0.2)
# OpenCV không có fspecial nên dùng scipy hoặc tự tạo kernel
gaussianlp = cv2.GaussianBlur(marked, (5, 5), 0.2)
cv2.imwrite('gaussianlp.bmp', gaussianlp)

# 3. Cắt vùng ảnh (gán giá trị rất sáng)
cutted = marked.copy()
cutted[0:64, 0:400] = 255  # 255 = trắng, tương đương giá trị lớn như 512 trong uint8
cv2.imwrite('cutted.bmp', cutted)

# 4. Xoay ảnh 10 độ (bilinear + crop)
rotated = rotate(marked, 10, reshape=False, order=1)  # order=1 tương đương bilinear
rotated = np.clip(rotated, 0, 255).astype(np.uint8)
cv2.imwrite('rotated.bmp', rotated)

# 5. Thu nhỏ ảnh (scale 0.5)
scaled = cv2.resize(marked, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imwrite('scaled.bmp', scaled)

# 6. Nén JPEG với chất lượng 50
cv2.imwrite('compressed.jpg', marked, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
