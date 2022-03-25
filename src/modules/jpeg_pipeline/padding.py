"""------------DESTRUCTIVE COMPRESSION OF matrix - PARTIAL JPEG CODEC------------
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


def apply_padding(matrix: np.ndarray, wanted_rows: int, wanted_cols: int, interpolation_type: int = None):
    """
    Applies padding to the matrix.
    :param matrix: the matrix to pad.
    :param wanted_rows: number of rows that the matrix height must be multiple of.
    :param wanted_cols: number of columns that the matrix height must be multiple of.
    :param interpolation_type: the chosen interpolation (e.g. LINEAR; CUBIC; AREA).
    :return: the padded matrix.
    """
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]

    if interpolation_type is not None:
        remaining_rows = int()
        remaining_cols = int()

        if wanted_rows != 0:
            remaining_rows = wanted_rows - n_rows % wanted_rows
        if wanted_cols != 0:
            remaining_cols = wanted_cols - n_cols % wanted_cols

        matrix = cv2.resize(matrix, (n_cols + remaining_cols, n_rows + remaining_rows), interpolation_type)
    else:
        if wanted_rows != 0:
            remaining = n_rows % wanted_rows

            if remaining != 0:
                remaining = wanted_rows - remaining
                last_row = matrix[n_rows - 1]
                rows_to_add = np.repeat([last_row], remaining, axis=0)
                matrix = np.vstack((matrix, rows_to_add))

        if wanted_cols != 0:
            remaining = n_cols % wanted_cols

            if remaining != 0:
                remaining = wanted_cols - remaining
                last_col = matrix[:, [-1]]
                matrix = np.hstack((matrix, np.tile(last_col, (remaining, 1))))

    return matrix


def inverse_padding(padded_matrix: np.ndarray, original_rows: int, original_cols: int, interpolation_type: int = None):
    """
    Reverses the padding.
    :param padded_matrix: the padded matrix to unpad.
    :param original_rows: number of original rows.
    :param original_cols: number of original columns.
    :param interpolation_type: the chosen interpolation (e.g. LINEAR; CUBIC; AREA).
    :return: the original matrix.
    """
    n_rows = padded_matrix.shape[0]
    n_cols = padded_matrix.shape[1]

    if interpolation_type is not None:
        return cv2.resize(padded_matrix, (original_cols, original_rows), interpolation_type)
    else:
        n_rows_to_delete = n_rows - original_rows
        if n_rows_to_delete != 0:
            rows_to_delete = np.arange(n_rows - n_rows_to_delete - 1, n_rows - 1)
            padded_matrix = np.delete(padded_matrix, rows_to_delete, axis=0)

        n_cols_to_delete = n_cols - original_cols
        if n_cols_to_delete != 0:
            cols_to_delete = np.arange(n_cols - n_cols_to_delete - 1, n_cols - 1)
            padded_matrix = np.delete(padded_matrix, cols_to_delete, axis=1)

        return padded_matrix
