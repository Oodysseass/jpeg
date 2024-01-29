from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import functions as fn


# quantization tables
y_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

c_table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]])

# load image
image = Image.open('baboon.png')
image = np.array(image)

# make dims multiple of 8
dim_M = image.shape[0] % 8
dim_N = image.shape[1] % 8
if dim_M != 0:
  image = image[:-dim_M, :, :]
if dim_N != 0:
  image = image[:, :-dim_N, :]

# convert to ycrcb
imageY, imageCr, imageCb = fn.convert2ycrcb(image, [4, 2, 0])

# convert to dct blocks
blocks_y = {}
blocks_cr = {}
blocks_cb = {}
for i in range(imageY.shape[0] // 8):
  blocks_y[i] = {}
  for j in range(imageY.shape[1] // 8):
    blocks_y[i][j] = fn.blockDCT(imageY[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

for i in range(imageCr.shape[0] // 8):
  blocks_cr[i] = {}
  blocks_cb[i] = {}
  for j in range(imageCr.shape[1] // 8):
    blocks_cr[i][j] = fn.blockDCT(imageCr[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
    blocks_cb[i][j] = fn.blockDCT(imageCb[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

# quantize
for i in range(imageY.shape[0] // 8):
  for j in range(imageY.shape[1] // 8):
    blocks_y[i][j] = fn.quantizeJPEG(blocks_y[i][j], y_table, 0.5)

for i in range(imageCr.shape[0] // 8):
  for j in range(imageCr.shape[1] // 8):
    blocks_cr[i][j] = fn.quantizeJPEG(blocks_cr[i][j], c_table, 0.5)
    blocks_cb[i][j] = fn.quantizeJPEG(blocks_cb[i][j], c_table, 0.5)

# encode
symbols_y = {}
symbols_cr = {}
symbols_cb = {}
symbols_y[0] = {}
symbols_y[0][0] = fn.runLength(symbols_y[0][0], 0)
for i in range(imageY.shape[0] // 8):
  symbols_y[i] = {}
  for j in range(imageY.shape[1] // 8):
    if i == 0 and j == 0:
      continue
    prev = [i, j - 1] if j > 0 else [i - 1, imageY.shape[1] // 8 - 1]
    blocks_y[i][j] = fn.runLength(blocks_y[i][j], blocks_y[prev[0]][prev[1]][0][1])

blocks_cr[0][0] = fn.runLength(blocks_cr[0][0], 0)
blocks_cb[0][0] = fn.runLength(blocks_cb[0][0], 0)
for i in range(imageCr.shape[0] // 8):
  for j in range(imageCr.shape[1] // 8):
    if i == 0 and j == 0:
      continue
    prev = [i, j - 1] if j > 0 else [i - 1, imageCr.shape[1] // 8 - 1]
    blocks_cr[i][j] = fn.runLength(blocks_cr[i][j], blocks_cr[prev[0]][prev[1]][0][1])
    blocks_cb[i][j] = fn.runLength(blocks_cb[i][j], blocks_cb[prev[0]][prev[1]][0][1])


## inverse
# decode
blocks_y[0][0] = fn.irunLength(blocks_y[0][0], 0)
for i in range(imageY.shape[0] // 8):
  for j in range(imageY.shape[1] // 8):
    if i == 0 and j == 0:
      continue
    prev = [i, j - 1] if j > 0 else [i - 1, imageY.shape[1] // 8 - 1]
    blocks_y[i][j] = fn.irunLength(blocks_y[i][j], blocks_y[prev[0]][prev[1]][0, 0])
print(blocks_y[0][0])
print(blocks_y[0][1])
print(blocks_y[0][2])

blocks_cr[0][0] = fn.irunLength(blocks_cr[0][0], 0)
blocks_cb[0][0] = fn.irunLength(blocks_cb[0][0], 0)
for i in range(imageCr.shape[0] // 8):
  for j in range(imageCr.shape[1] // 8):
    if i == 0 and j == 0:
      continue
    prev = [i, j - 1] if j > 0 else [i - 1, imageCr.shape[1] // 8 - 1]
    blocks_cr[i][j] = fn.irunLength(blocks_cr[i][j], blocks_cr[prev[0]][prev[1]][0][0])
    blocks_cb[i][j] = fn.irunLength(blocks_cb[i][j], blocks_cb[prev[0]][prev[1]][0][0])

# dequantize
for i in range(imageY.shape[0] // 8):
  for j in range(imageY.shape[1] // 8):
    blocks_y[i][j] = fn.dequantizeJPEG(blocks_y[i][j], y_table, 0.5)

for i in range(imageCr.shape[0] // 8):
  for j in range(imageCr.shape[1] // 8):
    blocks_cr[i][j] = fn.dequantizeJPEG(blocks_cr[i][j], c_table, 0.5)
    blocks_cb[i][j] = fn.dequantizeJPEG(blocks_cb[i][j], c_table, 0.5)

# inverse dct
image_Y = np.zeros(imageY.shape)
image_Cr = np.zeros(imageCr.shape)
image_Cb = np.zeros(imageCb.shape)
for i in range(imageY.shape[0] // 8):
  for j in range(imageY.shape[1] // 8):
    image_Y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_y[i][j])

for i in range(imageCr.shape[0] // 8):
  for j in range(imageCr.shape[1] // 8):
    image_Cr[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cr[i][j])
    image_Cb[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cb[i][j])

# convert to rgb
imageRGB = fn.convert2rgb(image_Y, image_Cr, image_Cb, [4, 2, 0])


## display
imageRGB = imageRGB / 255.0
image = image / 255.0

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(imageRGB)
plt.title('ImageRGB')

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.title('Image')

plt.tight_layout()
plt.show()