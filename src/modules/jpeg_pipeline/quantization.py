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


import cv2
import numpy as np


# region Public Methods

def get_scale_factor(quality_factor: float):
    """
    Function that retrieves the scale factor based on quality factor.
    :param quality_factor: the quality factor from 0 to 100.
    :return: the scale factor.
    """
    if quality_factor >= 50:
        return (100 - quality_factor) / 50
    else:
        return 50 / quality_factor


def get_scaled_quantization_matrix(quality_factor: float, quantization_matrix: np.ndarray):
    """
    Function that retrieves the scaled quantization matrix.
    :param quality_factor: the quality factor of the matrix.
    :param quantization_matrix: the quantization matrix.
    :return: the final quantization matrix.
    """
    scale_factor = get_scale_factor(quality_factor)

    if scale_factor == 0:
        return np.ones((quantization_matrix.shape[0], quantization_matrix.shape[1]))

    scaled_quantization_matrix = np.round(quantization_matrix * scale_factor)
    scaled_quantization_matrix[scaled_quantization_matrix > 255] = 255
    scaled_quantization_matrix[scaled_quantization_matrix < 1] = 1

    return scaled_quantization_matrix


def apply_quantization(matrix: np.ndarray, quality_factor: float, quantization_matrix: np.ndarray):
    """
    Function to apply quantization.
    :param matrix: matrix to apply quantization.
    :param quality_factor: the quality factor to be applied.
    :param quantization_matrix: the quantization matrix.
    :return: the quantized original matrix.
    """

    if matrix.shape[2] == 8:
        q = get_scaled_quantization_matrix(quality_factor, quantization_matrix)
    else:
        q = cv2.resize(get_scaled_quantization_matrix(quality_factor, quantization_matrix),
                       (matrix.shape[2], matrix.shape[3]),
                       interpolation=cv2.INTER_CUBIC)

    return np.round(matrix / q)


def apply_inverse_quantization(matrix: np.ndarray, quality_factor: float, quantization_matrix: np.ndarray):
    """
    Function to apply inverse quantization.
    :param matrix: matrix to apply quantization.
    :param quality_factor: the quality factor to be applied.
    :param quantization_matrix: the quantization matrix.
    :return: the original matrix.
    """
    resized_q = cv2.resize(get_scaled_quantization_matrix(quality_factor, quantization_matrix),
                           (matrix.shape[2], matrix.shape[3]),
                           interpolation=cv2.INTER_CUBIC)

    return matrix * resized_q

# endregion Public Methods
