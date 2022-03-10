import numpy as np


def separate_channels(img):
    """
             Separates the rgb channels of an image.
             :param img: the image.
             :return: RGB channels.
    """

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
