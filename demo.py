from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from functions import convert2rgb, convert2ycrcb, blockDCT, iBlockDCT


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
imageY, imageCr, imageCb = convert2ycrcb(image, [4, 2, 0])

# convert to dct blocks
blocks_y = {}
blocks_cr = {}
blocks_cb = {}
for i in range(imageY.shape[0] // 8):
  blocks_y[i] = {}
  for j in range(imageY.shape[1] // 8):
    blocks_y[i][j] = blockDCT(imageY[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

for i in range(imageCr.shape[0] // 8):
  blocks_cr[i] = {}
  blocks_cb[i] = {}
  for j in range(imageCr.shape[1] // 8):
    blocks_cr[i][j] = blockDCT(imageCr[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
    blocks_cb[i][j] = blockDCT(imageCb[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])


## inverse jpeg
# inverse dct
image_Y = np.zeros(imageY.shape)
image_Cr = np.zeros(imageCr.shape)
image_Cb = np.zeros(imageCb.shape)
for i in range(imageY.shape[0] // 8):
  for j in range(imageY.shape[1] // 8):
    image_Y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = iBlockDCT(blocks_y[i][j])

for i in range(imageCr.shape[0] // 8):
  for j in range(imageCr.shape[1] // 8):
    image_Cr[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = iBlockDCT(blocks_cr[i][j])
    image_Cb[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = iBlockDCT(blocks_cb[i][j])

# convert to rgb
imageRGB = convert2rgb(imageY, imageCr, imageCb, [4, 2, 0])

# display
imageRGB = imageRGB / 255.0
plt.imshow(imageRGB)
plt.show()
