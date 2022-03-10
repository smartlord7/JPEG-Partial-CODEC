import numpy as np
from matplotlib import pyplot as plt


def separate_rgb(img, show_plots=False):
    """
             Separates the rgb channels of an image.
             :param img: the image.
             :param show_plots: flag that toggles if it plots the channels.
             :return: RGB channels matrix.
    """
    r, g, b = img.copy(), img.copy(), img.copy()
    r[:, :, (1, 2)] = 0
    g[:, :, (0, 2)] = 0
    b[:, :, (0, 1)] = 0

    if show_plots:
        img_rgb = np.concatenate((r, g, b))
        plt.figure()
        plt.title("Separate RGB")
        plt.imshow(img_rgb)
        plt.show()

    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def join_channels(c1, c2, c3):
    """
             Function that joins the RGB channels in one.
             :param c1: RGB Channel 1.
             :param c2: RGB Channel 2.
             :param c3: RGB Channel 3.
             :return: image that results in the of the 3 channels.
    """
    shape = (c1.shape[0], c2.shape[1], 3)
    image = np.zeros(shape)
    image[:, :, 0] = c1
    image[:, :, 1] = c2
    image[:, :, 2] = c3

    return image


def float_to_uint8(matrix):
    """
                Converts float to uint8.
                :param matrix: Matrix with the floats.
                :return: UINT8 converted matrix.
    """
    matrix = matrix.round()
    matrix[matrix > 255] = 255
    matrix[matrix < 0] = 0
    matrix = matrix.astype(np.uint8)

    return matrix
