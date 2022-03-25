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
from modules.util import float_to_uint8


def rgb_to_y_cb_cr(rgb: np.ndarray, y_cb_cr_matrix: np.ndarray):
    """
    Function that Converts RGB to YCbCr.
    :param rgb: RGB matrix.
    :param y_cb_cr_matrix: YCbCr default values matrix.
    :return: YCbCr converted matrix.
    """
    y_cb_cr = rgb.dot(y_cb_cr_matrix.T)
    y_cb_cr[:, :, [1, 2]] += 128

    return y_cb_cr


def y_cb_cr_to_rgb(y_cb_cr: np.ndarray, y_cb_cr_inverse_matrix: np.ndarray):
    """
    Function that converts RGB to YCbCr.
    :param y_cb_cr: YCbCr matrix.
    :param y_cb_cr_inverse_matrix: YCbCr inverse default values matrix.
    :return: RGB converted matrix.
    """
    y_cb_cr[:, :, [1, 2]] -= 128
    rgb = y_cb_cr.dot(y_cb_cr_inverse_matrix.T)
    rgb = float_to_uint8(rgb)

    return rgb
