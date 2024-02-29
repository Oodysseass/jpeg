from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from jpeg import JPEGenc

image1 = Image.open('baboon.png')
image1 = np.array(image1)
image2 = Image.open('lena_color_512.png')
image2 = np.array(image2)
q = [0.1, 0.3, 0.6, 1, 2, 5, 10]
mse = []

plt.figure(figsize=(15, 10))
for i in range(len(q)):
  jpeg = JPEGenc.JPEGencode(image1, [4, 4, 4], q[i])
  img = JPEGenc.JPEGdecode(jpeg)

  plt.subplot(2, 4, i+1)
  plt.imshow(img)
  plt.title(f'qScale={q[i]}')
plt.tight_layout()
plt.savefig('5.png')
plt.show()

plt.figure(figsize=(15, 10))
for i in range(len(q)):
  jpeg = JPEGenc.JPEGencode(image2, [4, 4, 4], q[i])
  img = JPEGenc.JPEGdecode(jpeg)
  plt.subplot(2, 4, i+1)
  plt.imshow(img)
  plt.title(f'qScale={q[i]}')
plt.tight_layout()
plt.savefig('6.png')
plt.show()
