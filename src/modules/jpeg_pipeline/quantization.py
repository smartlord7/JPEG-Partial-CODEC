import numpy as np


def get_scale_factor(quality_factor):
    """
    Function that retrieves the scale factor of the image.
    :param quality_factor: the quality factor of the image.
    :return: the scale factor of the image.
    """
    if quality_factor >= 50:
        return (100 - quality_factor) / 50
    else:
        return 50 / quality_factor


def get_scaled_quantization_matrix(quality_factor, quantization_matrix):
    """
        Function that retrieves the scaled quantization matrix.
        :param quality_factor: the quality factor of the image.
        :param quantization_matrix: the quantization matrix.
        :return: the final quantization matrix.
    """
    scale_factor = get_scale_factor(quality_factor)

    if scale_factor == 0:
        return np.ones((quantization_matrix.shape[0], quantization_matrix.shape[1]))

    scaled_quantization_matrix = quantization_matrix * scale_factor
    scaled_quantization_matrix[quantization_matrix * scale_factor > 255] = 255

    return scaled_quantization_matrix


def apply_quantization(matrix, quality_factor, quantization_matrix):
    """ Function to apply quantization """
    return np.round(matrix / get_scaled_quantization_matrix(quality_factor, quantization_matrix))


def apply_inverse_quantization(matrix, quality_factor, quantization_matrix):
    """ Function to apply inverse quantization """
    return matrix * get_scaled_quantization_matrix(quality_factor, quantization_matrix)


