"""------------DESTRUCTIVE COMPRESSION OF IMAGE - PARTIAL JPEG CODEC------------
University of Coimbra
Degree in Computer Science and Engineering
Multimedia
3rd year, 2nd semester
Authors:
Rui Bernardo Lopes Rodrigues, 2019217573, uc2019217573@student.uc.pt
Sancho Amaral Simões, 2019217590, uc2019217590@student.uc.pt
Tiago Filipe Santa Ventura, 2019243695, uc2019243695@student.uc.pt
Coimbra, 23rd March 2022
---------------------------------------------------------------------------"""

import numpy as np
from modules.image import generate_linear_colormap

ORIGINAL_IMAGE_DIRECTORY = '\\resources\\img\\original_img\\'
ORIGINAL_IMAGE_EXTENSION = '.bmp'
COMPRESSED_IMAGE_DIRECTORY = '\\resources\\img\\jpeg_compressed_img'

JPEG_QUALITY_RATES = [25, 50, 75]
JPEG_QUANTIZATION_Y = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
               [12, 12, 14, 19,  26,  58,  60,  55],
               [14, 13, 16, 24,  40,  57,  69,  56],
               [14, 17, 22, 29,  51,  87,  80,  62],
               [18, 22, 37, 56,  68, 109, 103,  77],
               [24, 35, 55, 64,  81, 104, 113,  92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103,  99]])
JPEG_QUANTIZATION_CB_CR = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])


Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
Y_CB_CR_MATRIX_INVERSE = np.linalg.inv(Y_CB_CR_MATRIX)
GREY_CMAP_LIST = [(0, 0, 0), (1, 1, 1)]
RED_CMAP_LIST = [(0, 0, 0), (1, 0, 0)]
GREEN_CMAP_LIST = [(0, 0, 0), (0, 1, 0)]
BLUE_CMAP_LIST = [(0, 0, 0), (0, 0, 1)]

GREY_CMAP = generate_linear_colormap(GREY_CMAP_LIST)
RED_CMAP = generate_linear_colormap(RED_CMAP_LIST)
GREEN_CMAP = generate_linear_colormap(GREEN_CMAP_LIST)
BLUE_CMAP = generate_linear_colormap(BLUE_CMAP_LIST)

IMAGE_SIZE_DIVISOR = 16
