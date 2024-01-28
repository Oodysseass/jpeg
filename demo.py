from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from functions import convert2rgb, convert2ycrcb


# load image
image = Image.open('baboon.png')
image = np.array(image) / 255.0

# make dims multiple of 8
dim_M = image.shape[0] % 8
dim_N = image.shape[1] % 8
if dim_M != 0:
  image = image[:-dim_M, :, :]
if dim_N != 0:
  image = image[:, :-dim_N, :]

# convert to ycrcb
imageY, imageCr, imageCb = convert2ycrcb(image, [4, 2, 0])

# convert to rgb
imageRGB = convert2rgb(imageY, imageCr, imageCb, [4, 2, 0])

plt.imshow(imageRGB)
plt.show()