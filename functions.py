import numpy as np


def convert2ycrcb(imageRGB, subimg):
  # transformation matrix 2ycrcb
  T = np.array([[0.299, 0.587, 0.114],
                [0.147, 0.289, 0.436],
                [0.615, -0.515, -0.1]])

  # calculate ycrcb
  imageY = T[0, 0] * imageRGB[:, :, 0] \
         + T[0, 1] * imageRGB[:, :, 1] \
         + T[0, 2] * imageRGB[:, :, 2]

  imageCr = T[1, 0] * imageRGB[:, :, 0] \
          + T[1, 1] * imageRGB[:, :, 1] \
          + T[1, 2] * imageRGB[:, :, 2]

  imageCb = T[2, 0] * imageRGB[:, :, 0] \
          + T[2, 1] * imageRGB[:, :, 1] \
          + T[2, 2] * imageRGB[:, :, 2]

  # downsample
  if subimg[2] == 2:
    imageCr = imageCr[:, 0:-1:2]
    imageCb = imageCb[:, 0:-1:2]
  elif subimg[2] == 0:
    imageCr = imageCr[0:-1:2, 0:-1:2]
    imageCb = imageCb[0:-1:2, 0:-1:2]

  return imageY, imageCr, imageCb


def convert2rgb(imageY, imageCr, imageCb, subimg):
  # inverse matrix 2RGB
  T = np.array([[0.299, 0.587, 0.114],
                [0.147, 0.289, 0.436],
                [0.615, -0.515, -0.1]])
  T = np.linalg.inv(T)

  # upsample
  imageCr_up = np.zeros(imageY.shape)
  imageCb_up = np.zeros(imageY.shape)
  if subimg[2] == 2:
    imageCr_up[:, 0::2] = imageCr
    imageCb_up[:, 0::2] = imageCb

    imageCr_up[:, 1::2] = imageCr_up[:, 0::2]
    imageCb_up[:, 1::2] = imageCb_up[:, 0::2]
  elif subimg[2] == 0:
    imageCr_up[0::2, 0::2] = imageCr
    imageCb_up[0::2, 0::2] = imageCb

    imageCr_up[0::2, 1::2] = imageCr_up[0::2, 0::2]
    imageCr_up[1::2, :] = imageCr_up[0::2, :]
    imageCb_up[0::2, 1::2] = imageCb_up[0::2, 0::2]
    imageCb_up[1::2, :] = imageCb_up[0::2, :]

  # reverse transform
  red = T[0, 0] * imageY  \
      + T[0, 1] * imageCr_up \
      + T[0, 2] * imageCb_up
  
  green = T[1, 0] * imageY  \
        + T[1, 1] * imageCr_up \
        + T[1, 2] * imageCb_up

  blue = T[2, 0] * imageY  \
       + T[2, 1] * imageCr_up \
       + T[2, 2] * imageCb_up

  # create image
  imageRGB = np.array([red, green, blue])
  imageRGB = np.transpose(imageRGB, (1, 2, 0))

  return np.clip(imageRGB, 0, 1)
