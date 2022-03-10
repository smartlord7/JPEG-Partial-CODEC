import numpy as np
from modules.image import generate_linear_colormap

Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
GREY_CMAP_LIST = [(0, 0, 0), (1, 1, 1)]
RED_CMAP_LIST = [(0, 0, 0), (1, 0, 0)]
GREEN_CMAP_LIST = [(0, 0, 0), (0, 1, 0)]
BLUE_CMAP_LIST = [(0, 0, 0), (0, 0, 1)]

GREY_CMAP = generate_linear_colormap(GREY_CMAP_LIST)
RED_CMAP = generate_linear_colormap(RED_CMAP_LIST)
GREEN_CMAP = generate_linear_colormap(GREEN_CMAP_LIST)
BLUE_CMAP = generate_linear_colormap(BLUE_CMAP_LIST)

JPEG_QUALITY_RATES = [25, 50, 75]

ORIGINAL_IMAGE_DIRECTORY = '\\resources\\img\\original_img\\'
ORIGINAL_IMAGE_EXTENSION = '.bmp'
COMPRESSED_IMAGE_DIRECTORY = '\\jpeg_compressed_img'