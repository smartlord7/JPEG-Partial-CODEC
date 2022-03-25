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


import os
import numpy as np


def separate_channels(img: np.ndarray):
    """
    Separates the rgb channels of an image.
    :param img: the image containing the channels to separate.
    :return: RGB channels.
    """

    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def join_channels(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray):
    """
    Function that joins the RGB channels in one.
    :param c1: RGB Channel 1.
    :param c2: RGB Channel 2.
    :param c3: RGB Channel 3.
    :return: image that results in the junction of the 3 channels.
    """
    shape = (c1.shape[0], c2.shape[1], 3)
    image = np.zeros(shape)
    image[:, :, 0] = c1
    image[:, :, 1] = c2
    image[:, :, 2] = c3

    return image


def float_to_uint8(matrix: np.ndarray):
    """
    Converts float to uint8.
    :param matrix: Matrix with the floats.
    :return: uint8 converted matrix.
    """
    matrix = matrix.round()
    matrix[matrix > 255] = 255
    matrix[matrix < 0] = 0
    matrix = matrix.astype(np.uint8)

    return matrix


def out(output_file, string):
    """
    Function the prints a string to stdout and to an output file.
    :param output_file: The file in which the string will be dumped.
    :param string: the string to printed/written.
    """
    print(string)
    output_file.write(string + "\n")
    

def mkdir_if_not_exists(path):
    """
    Function that creates a directory if not exists.
    :param path: the directory path.
    """
    if not os.path.exists(path):
        os.mkdir(path)
