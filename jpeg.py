import numpy as np

import functions as fn
from helpers import dc_lum, dc_chrom, ac_lum, ac_chrom, default_l, default_c


# class determining an encoded block of the image
class JPEGblock:
  def __init__(self, blkType, indHor, indVer, huffStream, imgeRec):
    self.blkType = blkType
    self.indHor = indHor
    self.indVer = indVer
    self.huffStream = huffStream
    self.imgeRec = imgeRec


# class with the general info of the encoded image
class JPEGenc:
  def __init__(self, qScale=1, \
               DCL=dc_lum, DCC=dc_chrom, \
               ACL=ac_lum, ACC=ac_chrom, \
               qTableL=default_l, qTableC=default_c):
    self.qScale = qScale
    self.DCL = DCL
    self.DCC = DCC
    self.ACL = ACL
    self.ACC = ACC
    self.qTableL = qTableL
    self.qTableC = qTableC

  def get_run_dc_lum(self, code):
    for key, value in self.DCL.items():
      if value == code:
        return key
    return None

  def get_run_dc_chrom(self, code):
    for key, value in self.DCC.items():
      if value == code:
        return key
    return None

  def get_run_ac_lum(self, code):
    for key, value in self.ACL.items():
      if value == code:
        return key
    return None

  def get_run_ac_chrom(self, code):
    for key, value in self.ACC.items():
      if value == code:
        return key
    return None

  def get_huffman_dc_lum(self, cat):
    return self.DCL[cat]
  
  def get_huffman_dc_chrom(self, cat):
    return self.DCC[cat]
  
  def get_huffman_ac_lum(self, cat):
    return self.ACL[cat]
  
  def get_huffman_ac_chrom(self, cat):
    return self.ACC[cat]
  
  def get_huffman(self, ac_dc, lum_chrom, cat):
    if ac_dc == 'ac':
      if lum_chrom == 'Y':
        return self.get_huffman_ac_lum(cat)
      else:
        return self.get_huffman_ac_chrom(cat)
    else:
      if lum_chrom == 'Y':
        return self.get_huffman_dc_lum(cat)
      else:
        return self.get_huffman_dc_chrom(cat)

  def get_symbol(self, ac_dc, lum_chrom, code):
    if ac_dc == 'ac':
      if lum_chrom == 'Y':
        return self.get_run_ac_lum(code)
      else:
        return self.get_run_ac_chrom(code)
    else:
      if lum_chrom == 'Y':
        return self.get_run_dc_lum(code)
      else:
        return self.get_run_dc_chrom(code)

  @staticmethod
  def JPEGencode(img, subimg, qScale, qTableL=default_l, qTableC=default_l):
    header = JPEGenc(qScale=qScale, qTableL=qTableL, qTableC=qTableC)

    # make dims multiple of 8
    dim_M = img.shape[0] % 8
    dim_N = img.shape[1] % 8
    if dim_M != 0:
      img = img[:-dim_M, :, :]
    if dim_N != 0:
      img = img[:, :-dim_N, :]

    ## jpeg
    # convert to ycrcb
    imageY, imageCr, imageCb = fn.convert2ycrcb(img, subimg)

    # convert to dct blocks
    M_y = imageY.shape[0] // 8
    N_y = imageY.shape[1] // 8
    M_c = imageCr.shape[0] // 8
    N_c = imageCr.shape[1] // 8
    blocks_y = {i: {} for i in range(M_y)}
    blocks_cr = {i: {} for i in range(M_c)}
    blocks_cb = {i: {} for i in range(M_c)}

    for i in range(M_y):
      for j in range(N_y):
        blocks_y[i][j] = fn.blockDCT(imageY[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

    for i in range(M_c):
      for j in range(N_c):
        blocks_cr[i][j] = fn.blockDCT(imageCr[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])
        blocks_cb[i][j] = fn.blockDCT(imageCb[i * 8:(i + 1) * 8, j * 8: (j + 1) * 8])

    # quantize
    for i in range(M_y):
      for j in range(N_y):
        blocks_y[i][j] = fn.quantizeJPEG(blocks_y[i][j], header.qTableL, qScale)

    for i in range(M_c):
      for j in range(N_c):
        blocks_cr[i][j] = fn.quantizeJPEG(blocks_cr[i][j], header.qTableC, qScale)
        blocks_cb[i][j] = fn.quantizeJPEG(blocks_cb[i][j], header.qTableC, qScale)

    # encode to run symbols
    symbols_y = {i: {} for i in range(M_y)}
    symbols_cr = {i: {} for i in range(M_c)}
    symbols_cb = {i: {} for i in range(M_c)}

    symbols_y[0][0] = fn.runLength(blocks_y[0][0], 0)
    for i in range(M_y):
      for j in range(N_y):
        if i == 0 and j == 0:
          continue
        prev = [i, j - 1] if j > 0 else [i - 1, N_y - 1]
        dc_pred = blocks_y[prev[0]][prev[1]][0, 0]
        symbols_y[i][j] = fn.runLength(blocks_y[i][j], dc_pred)

    symbols_cr[0][0] = fn.runLength(blocks_cr[0][0], 0)
    symbols_cb[0][0] = fn.runLength(blocks_cb[0][0], 0)
    for i in range(M_c):
      for j in range(N_c):
        if i == 0 and j == 0:
          continue
        prev = [i, j - 1] if j > 0 else [i - 1, N_c - 1]

        dc_pred = blocks_cr[prev[0]][prev[1]][0, 0]
        symbols_cr[i][j] = fn.runLength(blocks_cr[i][j], dc_pred)

        dc_pred = blocks_cb[prev[0]][prev[1]][0, 0]
        symbols_cb[i][j] = fn.runLength(blocks_cb[i][j], dc_pred)

    # huffman encoding
    # encode to run symbols
    huff_y = {i: {} for i in range(M_y)}
    huff_cr = {i: {} for i in range(M_c)}
    huff_cb = {i: {} for i in range(M_c)}
    for i in range(M_y):
      for j in range(N_y):
        huff_y[i][j] = fn.huffEnc(symbols_y[i][j], 'Y', header)

    for i in range(M_c):
      for j in range(N_c):
        huff_cr[i][j] = fn.huffEnc(symbols_cr[i][j], 'Cr', header)
        huff_cb[i][j] = fn.huffEnc(symbols_cb[i][j], 'Cb', header)
    
    
    ## build tuple
    jpeg = [header,]
    for i in range(M_y):
      for j in range(N_y):
        jpeg.append(JPEGblock('Y', i, j, huff_y[i][j], None))

    for i in range(M_c):
      for j in range(N_c):
        jpeg.append(JPEGblock('Cr', i, j, huff_cr[i][j], None))
        jpeg.append(JPEGblock('Cb', i, j, huff_cb[i][j], None))

    return jpeg

  @staticmethod
  def JPEGdecode(jpeg):
    header = jpeg[0]

    # retrieve blocks
    i = 1
    huff_y = {}
    while jpeg[i].blkType == 'Y':
      ii = jpeg[i].indHor
      jj = jpeg[i].indVer

      if jj == 0:
        huff_y[ii] = {}
      huff_y[ii][jj] = jpeg[i].huffStream

      i += 1
    M_y = ii + 1
    N_y = jj + 1

    huff_cr = {}
    huff_cb = {}
    while i < len(jpeg):
      ii = jpeg[i].indHor
      jj = jpeg[i].indVer

      if jj == 0:
        huff_cr[ii] = {}
        huff_cb[ii] = {}

      huff_cr[ii][jj] = jpeg[i].huffStream
      huff_cb[ii][jj] = jpeg[i + 1].huffStream

      i += 2
    M_c = ii + 1
    N_c = jj + 1

    # decode
    sym_y = {i: {} for i in range(M_y)}
    sym_cr = {i: {} for i in range(M_c)}
    sym_cb = {i: {} for i in range(M_c)}
    for i in range(M_y):
      for j in range(N_y):
        sym_y[i][j] = fn.huffDec(huff_y[i][j], 'Y', header)

    for i in range(M_c):
      for j in range(N_c):
        sym_cr[i][j] = fn.huffDec(huff_cr[i][j], 'Cr', header)
        sym_cb[i][j] = fn.huffDec(huff_cb[i][j], 'Cb', header)

    # decode run symbols
    blocks_y = {i: {} for i in range(M_y)}
    blocks_cr = {i: {} for i in range(M_c)}
    blocks_cb = {i: {} for i in range(M_c)}

    blocks_y[0][0] = fn.irunLength(sym_y[0][0], 0)
    for i in range(M_y):
      for j in range(N_y):
        if i == 0 and j == 0:
          continue
        prev = [i, j - 1] if j > 0 else [i - 1, N_y - 1]
        blocks_y[i][j] = fn.irunLength(sym_y[i][j], blocks_y[prev[0]][prev[1]][0, 0])

    blocks_cr[0][0] = fn.irunLength(sym_cr[0][0], 0)
    blocks_cb[0][0] = fn.irunLength(sym_cb[0][0], 0)
    for i in range(M_c):
      for j in range(N_c):
        if i == 0 and j == 0:
          continue
        prev = [i, j - 1] if j > 0 else [i - 1, N_c - 1]
        blocks_cr[i][j] = fn.irunLength(sym_cr[i][j], blocks_cr[prev[0]][prev[1]][0, 0])
        blocks_cb[i][j] = fn.irunLength(sym_cb[i][j], blocks_cb[prev[0]][prev[1]][0, 0])

    # dequantize
    for i in range(M_y):
      for j in range(N_y):
        blocks_y[i][j] = fn.dequantizeJPEG(blocks_y[i][j], header.qTableL, header.qScale)

    for i in range(M_c):
      for j in range(N_c):
        blocks_cr[i][j] = fn.dequantizeJPEG(blocks_cr[i][j], header.qTableC, header.qScale)
        blocks_cb[i][j] = fn.dequantizeJPEG(blocks_cb[i][j], header.qTableC, header.qScale)

    # inverse dct
    image_Y = np.zeros((M_y * 8, N_y * 8))
    image_Cr = np.zeros((M_c * 8, N_c * 8))
    image_Cb = np.zeros((M_c * 8, N_c * 8))
    for i in range(M_y):
      for j in range(N_y):
        image_Y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_y[i][j])

    for i in range(M_c):
      for j in range(N_c):
        image_Cr[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cr[i][j])
        image_Cb[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fn.iBlockDCT(blocks_cb[i][j])

    if M_y == M_c and N_y == N_c:
      subimg = [4,4,4]
    elif M_y == M_c and N_y == 2 * N_c:
      subimg = [4,2,2]
    elif M_y == 2 * M_c and N_y == 2 * N_c:
      subimg = [4,2,0]

    # convert to rgb
    image = fn.convert2rgb(image_Y, image_Cr, image_Cb, subimg)
    imgRec = image / 255.0
    
    return imgRec

  @staticmethod
  def get_category_dc(dc):
    if dc == 0:
        return 0
    elif abs(dc) <= 1:
        return 1
    elif abs(dc) <= 3:
        return 2
    elif abs(dc) <= 7:
        return 3
    elif abs(dc) <= 15:
        return 4
    elif abs(dc) <= 31:
        return 5
    elif abs(dc) <= 63:
        return 6
    elif abs(dc) <= 127:
        return 7
    elif abs(dc) <= 255:
        return 8
    elif abs(dc) <= 511:
        return 9
    elif abs(dc) <= 1023:
        return 10
    elif abs(dc) <= 2047:
        return 11
    else:
        print("No category for this difference")

  @staticmethod
  def get_category_ac(ac):
    if ac == 0:
        return 0
    elif abs(ac) <= 1:
        return 1
    elif abs(ac) <= 3:
        return 2
    elif abs(ac) <= 7:
        return 3
    elif abs(ac) <= 15:
        return 4
    elif abs(ac) <= 31:
        return 5
    elif abs(ac) <= 63:
        return 6
    elif abs(ac) <= 127:
        return 7
    elif abs(ac) <= 255:
        return 8
    elif abs(ac) <= 511:
        return 9
    elif abs(ac) <= 1023:
        return 10
    else:
        print("No category for this difference")
