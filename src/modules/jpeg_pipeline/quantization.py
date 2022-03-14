import numpy as np


def get_scale_factor(quality_factor):
    if quality_factor >= 50:
        return (100 - quality_factor) / 50
    else:
        return 50 / quality_factor


def get_scaled_quantization_matrix(quality_factor, quantization_matrix):
    scale_factor = get_scale_factor(quality_factor)

    if scale_factor == 0:
        return quantization_matrix

    return quantization_matrix * scale_factor


def apply_quantization(matrix, quality_factor, quantization_matrix):
    return np.round(matrix / get_scaled_quantization_matrix(quality_factor, quantization_matrix))


def apply_inverse_quantization(matrix, quality_factor, quantization_matrix):
    return matrix * get_scaled_quantization_matrix(quality_factor, quantization_matrix)


