from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from jpeg import JPEGenc

from functions import convert2ycrcb, blockDCT, quantizeJPEG, runLength
from helpers import mse, count_bits, create_table, \
                    entropy1, entropy2, default_l, default_c


# just for the sake of entropy.............................
def get_dct_runlength(image, subimg, q_scale):
  dim_M = image.shape[0] % 8
  dim_N = image.shape[1] % 8
  if dim_M != 0:
    image = image[:-dim_M, :, :]
  if dim_N != 0:
    image = image[:, :-dim_N, :]
  imageY, imageCr, imageCb = convert2ycrcb(image, subimg)
  M_y = imageY.shape[0] // 8
  N_y = imageY.shape[1] // 8
  M_c = imageCr.shape[0] // 8
  N_c = imageCr.shape[1] // 8
  blocks_y = np.zeros((M_y, N_y, 8, 8))
  blocks_cr = np.zeros((M_c, N_c, 8, 8))
  blocks_cb = np.zeros((M_c, N_c, 8, 8))
  for i in range(M_y):
    for j in range(N_y):
      blocks_y[i, j] = blockDCT(imageY[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
  for i in range(M_c):
    for j in range(N_c):
      blocks_cr[i, j] = blockDCT(imageCr[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
      blocks_cb[i, j] = blockDCT(imageCb[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
  for i in range(M_y):
    for j in range(N_y):
      blocks_y[i, j] = quantizeJPEG(blocks_y[i, j], default_l, q_scale)
  for i in range(M_c):
    for j in range(N_c):
      blocks_cr[i, j] = quantizeJPEG(blocks_cr[i, j], default_c, q_scale)
      blocks_cb[i, j] = quantizeJPEG(blocks_cb[i, j], default_c, q_scale)

  symbols_y = {i: {} for i in range(M_y)}
  symbols_cr = {i: {} for i in range(M_c)}
  symbols_cb = {i: {} for i in range(M_c)}
  symbols_y[0][0] = runLength(blocks_y[0, 0], 0)
  for i in range(M_y):
    for j in range(N_y):
      if i == 0 and j == 0:
        continue
      prev = [i, j - 1] if j > 0 else [i - 1, imageY.shape[1] // 8 - 1]
      dc_pred = blocks_y[prev[0]][prev[1]][0, 0]
      symbols_y[i][j] = runLength(blocks_y[i, j], dc_pred)
  symbols_cr[0][0] = runLength(blocks_cr[0, 0], 0)
  symbols_cb[0][0] = runLength(blocks_cb[0, 0], 0)
  for i in range(M_c):
    for j in range(N_c):
      if i == 0 and j == 0:
        continue
      prev = [i, j - 1] if j > 0 else [i - 1, N_c - 1]
      dc_pred = blocks_cr[prev[0]][prev[1]][0, 0]
      symbols_cr[i][j] = runLength(blocks_cr[i, j], dc_pred)
      dc_pred = blocks_cb[prev[0]][prev[1]][0, 0]
      symbols_cb[i][j] = runLength(blocks_cb[i, j], dc_pred)
  return blocks_y, blocks_cr, blocks_cb, symbols_y, symbols_cr, symbols_cb


## load images
image1 = Image.open('baboon.png')
image1 = np.array(image1)
image2 = Image.open('lena_color_512.png')
image2 = np.array(image2)

## mse and #bits for various qscales
q = [0.1, 0.3, 0.6, 1, 2, 5, 10]
errors = []
bits = []

plt.figure(figsize=(15, 10))
for i in range(len(q)):
  jpeg = JPEGenc.JPEGencode(image1, [4, 4, 4], q[i])
  img = JPEGenc.JPEGdecode(jpeg)
  errors.append(mse(img, image1 / 255.0))
  bits.append(count_bits(jpeg))

  plt.subplot(2, 4, i+1)
  plt.imshow(img)
  plt.title(f'qScale={q[i]}')

plt.tight_layout()
plt.savefig('5.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(q, errors)
plt.title('MSE over qScale')
plt.subplot(1, 2, 2)
plt.plot(q, bits)
plt.title('# bits over qScale')
plt.tight_layout()
plt.savefig('6.png')
plt.show()

plt.figure(figsize=(15, 10))
for i in range(len(q)):
  jpeg = JPEGenc.JPEGencode(image2, [4, 2, 2], q[i])
  img = JPEGenc.JPEGdecode(jpeg)
  errors[i] = mse(img, image1 / 255.0)
  bits[i] = count_bits(jpeg)

  plt.subplot(2, 4, i+1)
  plt.imshow(img)
  plt.title(f'qScale={q[i]}')

plt.tight_layout()
plt.savefig('7.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(q, errors)
plt.title('MSE over qScale')
plt.subplot(1, 2, 2)
plt.plot(q, bits)
plt.title('# bits over qScale')
plt.tight_layout()
plt.savefig('8.png')
plt.show()

## zero-out higher order freq of dct blocks
plt.figure(figsize=(20, 15))
high_freq = [20, 40, 50, 60, 63]
for i in range(len(high_freq)):
  q_table_l, q_table_c = create_table(9999999, high_freq[i])

  jpeg = JPEGenc.JPEGencode(image1, [4, 4, 4], 1, q_table_l, q_table_c)
  img = JPEGenc.JPEGdecode(jpeg)
  plt.subplot(2, 5, i+1)
  plt.imshow(img)
  plt.title(f'# high order acs={high_freq[i]}')

  jpeg = JPEGenc.JPEGencode(image2, [4, 4, 4], 1, q_table_l, q_table_c)
  img = JPEGenc.JPEGdecode(jpeg)
  plt.subplot(2, 5, i+6)
  plt.imshow(img)

plt.tight_layout()
plt.savefig('9.png')
plt.show()

## entropy
entropy_rgb1 = [entropy1(image1[:, :, 0].flatten()), \
                entropy1(image1[:, :, 1].flatten()), \
                entropy1(image1[:, :, 2].flatten())]
entropy_rgb2 = [entropy1(image2[:, :, 0].flatten()), \
                entropy1(image2[:, :, 1].flatten()), \
                entropy1(image2[:, :, 2].flatten())]

blocks_y1, blocks_cr1, blocks_cb1, run_y1, run_cr1, run_cb1 \
  = get_dct_runlength(image1, [4,2,2], 0.6)
blocks_y2, blocks_cr2, blocks_cb2, run_y2, run_cr2, run_cb2 \
  = get_dct_runlength(image2, [4,4,4], 5)
entropy_dct1 = [entropy1(blocks_y1.flatten()), \
                entropy1(blocks_cr1.flatten()), \
                entropy1(blocks_cb1.flatten())]
entropy_dct2 = [entropy1(blocks_y2.flatten()), \
                entropy1(blocks_cr2.flatten()), \
                entropy1(blocks_cb2.flatten())]

entropy_run1 = [entropy2(run_y1), \
                entropy2(run_cr1), \
                entropy2(run_cb1)]
entropy_run2 = [entropy2(run_y2), \
                entropy2(run_cr2), \
                entropy2(run_cb2)]

categories = ['R', 'G', 'B', 'DCT_Y', 'DCT_Cr', 'DCT_Cb', 'RL_Y', 'RL_Cr', 'RL_Cb']

entropies = entropy_rgb1 + entropy_dct1 + entropy_run1
plt.figure(figsize=(10, 6))
plt.bar(categories, entropies, color='skyblue')
plt.xlabel('Category')
plt.ylabel('Entropy')
plt.title('Entropies of RGB, DCT, and Run Length Symbols')
plt.xticks(rotation=45)
plt.savefig('10.png')
plt.show()

entropies = entropy_rgb2 + entropy_dct2 + entropy_run2
plt.figure(figsize=(10, 6))
plt.bar(categories, entropies, color='skyblue')
plt.xlabel('Category')
plt.ylabel('Entropy')
plt.title('Entropies of RGB, DCT, and Run Length Symbols')
plt.xticks(rotation=45)
plt.savefig('11.png')
plt.show()
