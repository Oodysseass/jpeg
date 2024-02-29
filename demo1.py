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

### first part
## image 1
# load image
image1 = Image.open('baboon.png')
image1 = np.array(image1)

# make dims multiple of 8
dim_M1 = image1.shape[0] % 8
dim_N1 = image1.shape[1] % 8
if dim_M1 != 0:
  image1 = image1[:-dim_M1, :, :]
if dim_N1 != 0:
  image1 = image1[:, :-dim_N1, :]

## image 2
# load image
image2 = Image.open('lena_color_512.png')
image2 = np.array(image2)

# make dims multiple of 8
dim_M2 = image2.shape[0] % 8
dim_N2 = image2.shape[1] % 8
if dim_M2 != 0:
  image2 = image2[:-dim_M2, :, :]
if dim_N2 != 0:
  image2 = image2[:, :-dim_N2, :]

## ycrcb
# convert to ycrcb
image_y1, image_cr1, image_cb1 = fn.convert2ycrcb(image1, [4, 2, 2])
image_y2, image_cr2, image_cb2 = fn.convert2ycrcb(image2, [4, 4, 4])

# revert
image_rgb1 = fn.convert2rgb(image_y1, image_cr1, image_cb1, [4, 2, 2])
image_rgb2 = fn.convert2rgb(image_y2, image_cr2, image_cb2, [4, 4, 4])

## plot
image_rgb1 = image_rgb1 / 255.0
image_rgb2 = image_rgb2 / 255.0

plt.subplot(1, 2, 1)
plt.imshow(image_rgb1)
plt.title('Image 1 recovered')

plt.subplot(1, 2, 2)
plt.imshow(image1)
plt.title('Image 1 original')

plt.suptitle("RGB <-> YCrCb")
plt.tight_layout()
plt.savefig('1.png', bbox_inches='tight')
plt.show()

plt.subplot(1, 2, 1)
plt.imshow(image_rgb2)
plt.title('Image 2 recovered')

plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.title('Image 2 original')

plt.suptitle("RGB <-> YCrCb")
plt.tight_layout()
plt.savefig('2.png', bbox_inches='tight')
plt.show()


### second part
## image 1
# convert to dct blocks
M_y1 = image_y1.shape[0] // 8
N_y1 = image_y1.shape[1] // 8
M_c1 = image_cr1.shape[0] // 8
N_c1 = image_cr1.shape[1] // 8
blocks_y1 = {i: {} for i in range(M_y1)}
blocks_cr1 = {i: {} for i in range(M_c1)}
blocks_cb1 = {i: {} for i in range(M_c1)}

for i in range(M_y1):
  for j in range(N_y1):
    blocks_y1[i][j] = fn.blockDCT(image_y1[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

for i in range(M_c1):
  for j in range(N_c1):
    blocks_cr1[i][j] = fn.blockDCT(image_cr1[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
    blocks_cb1[i][j] = fn.blockDCT(image_cb1[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

# quantize
for i in range(M_y1):
  for j in range(N_y1):
    blocks_y1[i][j] = fn.quantizeJPEG(blocks_y1[i][j], y_table, 0.6)

for i in range(M_c1):
  for j in range(N_c1):
    blocks_cr1[i][j] = fn.quantizeJPEG(blocks_cr1[i][j], c_table, 0.6)
    blocks_cb1[i][j] = fn.quantizeJPEG(blocks_cb1[i][j], c_table, 0.6)

## image 2
M_y2 = image_y2.shape[0] // 8
N_y2 = image_y2.shape[1] // 8
M_c2 = image_cr2.shape[0] // 8
N_c2 = image_cr2.shape[1] // 8
blocks_y2 = {i: {} for i in range(M_y2)}
blocks_cr2 = {i: {} for i in range(M_c2)}
blocks_cb2 = {i: {} for i in range(M_c2)}

for i in range(M_y2):
  for j in range(N_y2):
    blocks_y2[i][j] = fn.blockDCT(image_y2[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

for i in range(M_c2):
  for j in range(N_c2):
    blocks_cr2[i][j] = fn.blockDCT(image_cr2[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
    blocks_cb2[i][j] = fn.blockDCT(image_cb2[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

for i in range(M_y2):
  for j in range(N_y2):
    blocks_y2[i][j] = fn.quantizeJPEG(blocks_y2[i][j], y_table, 5)

for i in range(M_c2):
  for j in range(N_c2):
    blocks_cr2[i][j] = fn.quantizeJPEG(blocks_cr2[i][j], c_table, 5)
    blocks_cb2[i][j] = fn.quantizeJPEG(blocks_cb2[i][j], c_table, 5)

## revert
# dequantize
for i in range(M_y1):
  for j in range(N_y1):
    blocks_y1[i][j] = fn.dequantizeJPEG(blocks_y1[i][j], y_table, 0.6)

for i in range(M_c1):
  for j in range(N_c1):
    blocks_cr1[i][j] = fn.dequantizeJPEG(blocks_cr1[i][j], c_table, 0.6)
    blocks_cb1[i][j] = fn.dequantizeJPEG(blocks_cb1[i][j], c_table, 0.6)

# inverse dct
image_Y1 = np.zeros(image_y1.shape)
image_Cr1 = np.zeros(image_cr1.shape)
image_Cb1 = np.zeros(image_cb1.shape)
for i in range(image_y1.shape[0] // 8):
  for j in range(image_y1.shape[1] // 8):
    image_Y1[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_y1[i][j])

for i in range(image_cr1.shape[0] // 8):
  for j in range(image_cr1.shape[1] // 8):
    image_Cr1[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cr1[i][j])
    image_Cb1[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cb1[i][j])

# convert to rgb
image_rgb1 = fn.convert2rgb(image_Y1, image_Cr1, image_Cb1, [4, 2, 2])

## image 2
for i in range(M_y2):
  for j in range(N_y2):
    blocks_y2[i][j] = fn.dequantizeJPEG(blocks_y2[i][j], y_table, 5)

for i in range(M_c2):
  for j in range(N_c2):
    blocks_cr2[i][j] = fn.dequantizeJPEG(blocks_cr2[i][j], c_table, 5)
    blocks_cb2[i][j] = fn.dequantizeJPEG(blocks_cb2[i][j], c_table, 5)

# inverse dct
image_Y2 = np.zeros(image_y2.shape)
image_Cr2 = np.zeros(image_cr2.shape)
image_Cb2 = np.zeros(image_cb2.shape)
for i in range(image_y2.shape[0] // 8):
  for j in range(image_y2.shape[1] // 8):
    image_Y2[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_y2[i][j])

for i in range(image_cr2.shape[0] // 8):
  for j in range(image_cr2.shape[1] // 8):
    image_Cr2[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cr2[i][j])
    image_Cb2[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cb2[i][j])

# convert to rgb
image_rgb2 = fn.convert2rgb(image_Y2, image_Cr2, image_Cb2, [4, 4, 4])

## plot
image_rgb1 = image_rgb1 / 255.0
image_rgb2 = image_rgb2 / 255.0

plt.subplot(1, 2, 1)
plt.imshow(image_rgb1)
plt.title('Image 1 recovered')

plt.subplot(1, 2, 2)
plt.imshow(image1)
plt.title('Image 1 original')

plt.suptitle("Quantization and inverse")
plt.tight_layout()
plt.savefig('3.png', bbox_inches='tight')
plt.show()

plt.subplot(1, 2, 1)
plt.imshow(image_rgb2)
plt.title('Image 2 recovered')

plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.title('Image 2 original')

plt.suptitle("Quantization and inverse")
plt.tight_layout()
plt.savefig('4.png', bbox_inches='tight')
plt.show()
