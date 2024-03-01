from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from jpeg import JPEGenc


def mse(img1, img2):
  return np.mean(np.square(img1 - img2))

def count_bits(jpeg):
  sum = 0
  for i in range(1, len(jpeg)):
    sum += len(jpeg[i].huffStream)
  return sum


image1 = Image.open('baboon.png')
image1 = np.array(image1)
image2 = Image.open('lena_color_512.png')
image2 = np.array(image2)
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
  jpeg = JPEGenc.JPEGencode(image2, [4, 4, 4], q[i])
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
