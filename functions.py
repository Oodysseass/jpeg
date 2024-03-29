import numpy as np
import cv2 as cv

from helpers import twos_complement, inverse_twos_complement

TYPE = np.int64
NUM_BITS = 64


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

    imageCr_up[:, 1:-1:2] = (imageCr_up[:, 0:-2:2] + imageCr_up[:, 2::2]) / 2
    imageCr_up[:, -1] = imageCr_up[:, -2]

    imageCb_up[:, 1:-1:2] = (imageCb_up[:, 0:-2:2] + imageCb_up[:, 2::2]) / 2
    imageCb_up[:, -1] = imageCb_up[:, -2]
  elif subimg[2] == 0:
    imageCr_up[0::2, 0::2] = imageCr
    imageCb_up[0::2, 0::2] = imageCb

    imageCr_up[0::2, 1:-1:2] = (imageCr_up[0::2, 0:-2:2] + imageCr_up[0::2, 2::2]) / 2
    imageCr_up[0::2, -1] = imageCr_up[0::2, -2]
    imageCr_up[1:-1:2, :] = (imageCr_up[0:-2:2, :] + imageCr_up[2::2, :]) / 2
    imageCr_up[-1, :] = imageCr_up[-2, :]

    imageCb_up[0::2, 1:-1:2] = (imageCb_up[0::2, 0:-2:2] + imageCb_up[0::2, 2::2]) / 2
    imageCb_up[0::2, -1] = imageCb_up[0::2, -2]
    imageCb_up[1:-1:2, :] = (imageCb_up[0:-2:2, :] + imageCb_up[2::2, :]) / 2
    imageCb_up[-1, :] = imageCb_up[-2, :]
  else:
    imageCr_up = imageCr
    imageCb_up = imageCb

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

  #return np.clip(imageRGB, 0, 1)
  return imageRGB


# shift dct and inverse
def blockDCT(block):
  block = block - 128
  cv.dct(block, block, 0)
  return block

def iBlockDCT(block):
  cv.dct(block, block, 1)
  block = block + 128
  return block


# quantize blocks
def quantizeJPEG(dctBlock, qTable, qScale):
  q = qTable * qScale
  return (dctBlock / q).round()

def dequantizeJPEG(qBlock, qTable, qScale):
  q = qTable * qScale
  return qBlock * q


# encode run symbols
def runLength(qBlock, DCpred):
  # init with dc
  runSymbols = []
  runSymbols.append([0, qBlock[0][0] - DCpred])

  # start from (0, 1)
  i, j = 0, 1
  i_step = 1
  j_step = -1
  zero_counter = 0
  direction = True
  while i < 8 and j < 8:
    # raise counter or append based on current value
    if qBlock[i][j] == 0:
      zero_counter += 1
    else:
      runSymbols.append([zero_counter, qBlock[i][j]])
      zero_counter = 0

    # if a diagonal ended fix stepping
    if direction:
      if i == 0 and i_step == -1:
        i_step = 1
        j_step = -1
        j += 1
        continue
      elif j == 0 and j_step == -1:
        i_step = -1
        j_step = 1
        if i < 7:
          i += 1
        else:
          j += 1
          direction = False
        continue
    else:
      if i == 7 and i_step == 1:
        i_step = -1
        j_step = 1
        j += 1
        continue
      elif j == 7 and j_step == 1:
        i_step = 1
        j_step = -1
        i += 1
        continue

    i += i_step
    j += j_step

  if zero_counter != 0:
    runSymbols.append([zero_counter - 1, 0])

  return runSymbols

# decode run symbols
def irunLength(runSymbols, DCpred):
  qBlock = np.zeros((8, 8))
  qBlock[0, 0] = runSymbols[0][1] + DCpred

  # start from (0, 1)
  i, j = 0, 1
  i_step = 1
  j_step = -1
  symbol_counter = 1
  direction = True
  while i < 8 and j < 8 and symbol_counter < len(runSymbols):
    # add zero or dct value based on current run symbol
    if runSymbols[symbol_counter][0] == 0:
      qBlock[i][j] = runSymbols[symbol_counter][1]
      symbol_counter += 1
    else:
      runSymbols[symbol_counter][0] -= 1
      qBlock[i][j] = 0

    # if a diagonal ended fix stepping
    if direction:
      if i == 0 and i_step == -1:
        i_step = 1
        j_step = -1
        j += 1
        continue
      elif j == 0 and j_step == -1:
        i_step = -1
        j_step = 1
        if i < 7:
          i += 1
        else:
          j += 1
          direction = False
        continue
    else:
      if i == 7 and i_step == 1:
        i_step = -1
        j_step = 1
        j += 1
        continue
      elif j == 7 and j_step == 1:
        i_step = 1
        j_step = -1
        i += 1
        continue

    i += i_step
    j += j_step

  return qBlock


# huffman encoding
def huffEnc(runSymbols, blk_type, header):
  ## preprocess runSymbols
  # cut symbols with too many preceding zeros
  length = len(runSymbols)
  i = 0
  while i < length:
    if runSymbols[i][0] > 15 and runSymbols[i][1] != 0:
      runSymbols.insert(i, [15, 0])
      runSymbols[i + 1][0] -= 15
      length = len(runSymbols)
    i += 1

  # if last symbol is just zeros append special symbol
  if runSymbols[-1][1] == 0:
    runSymbols[-1] = [0, 0]

  runSymbols = np.array(runSymbols).astype(TYPE)

  ## array with symbols
  huff_stream = np.empty(len(runSymbols), dtype='object')

  # get dc category
  category = header.get_category_dc(runSymbols[0][1])
  # get SSSS of huffman code
  huff_stream[0] = header.get_huffman('dc', blk_type, category)
  # append LSB of value
  if category != 0:
    if runSymbols[0][1] > 0:
      bits = bin(runSymbols[0][1])[2:]
      lsb = bits[-category:]
    else:
      val = runSymbols[0][1] - 1
      comp = twos_complement(val, NUM_BITS)
      lsb = comp[-category:]
    huff_stream[0] = huff_stream[0] + lsb

  # do the same for acs
  for i in range(1, len(runSymbols)):
    prec_zer = runSymbols[i][0]
    category = header.get_category_ac(runSymbols[i][1])
    huff_stream[i] = header.get_huffman('ac', blk_type, (prec_zer, category))

    if category != 0:
      if runSymbols[i][1] > 0:
        bits = bin(runSymbols[i][1])[2:]
        lsb = bits[-category:]
      else:
        val = runSymbols[i][1] - 1
        comp = twos_complement(val, NUM_BITS)
        lsb = comp[-category:]
      huff_stream[i] = huff_stream[i] + lsb

  return huff_stream

# huffman decoding
def huffDec(huffStream, blk_type, header):
  run_symbols = []

  # find category
  code = huffStream[0]
  i = 0
  symbol = code[:2]
  category = header.get_symbol('dc', blk_type, symbol)
  while category is None:
    i += 1
    symbol = code[:2+i]
    category = header.get_symbol('dc', blk_type, symbol)

  # get value
  bin_value = code[-category:]
  value = 0
  if category != 0:
    if bin_value[0] == '1':
      value = int(bin_value, 2)
    else:
      value = inverse_twos_complement(bin_value) + 1

  run_symbols.append([0, value])

  # ac
  for i in range(1, len(huffStream)):
    code = huffStream[i]
    j = 0
    symbol = code[:2]
    category = header.get_symbol('ac', blk_type, symbol)
    while category is None:
      j += 1
      symbol = code[:2+j]
      category = header.get_symbol('ac', blk_type, symbol)

    prec_zeros, num_bits = category[0], category[1]

    # get value
    bin_value = code[-num_bits:]
    value = 0
    if num_bits != 0:
      if bin_value[0] == '1':
        value = int(bin_value, 2)
      else:
        value = inverse_twos_complement(bin_value) + 1

    run_symbols.append([prec_zeros, value])

  return run_symbols
