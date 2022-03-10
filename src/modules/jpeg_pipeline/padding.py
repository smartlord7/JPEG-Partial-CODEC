import numpy as np


def apply_padding(image, wanted_rows, wanted_cols):
    """
                            Applies padding to the image.
                            :param image: the image to pad.
                            :param wanted_rows: number of rows to pad.
                            :param wanted_cols: number of columns to pad.
                            :return: the image with padding.
    """
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    remaining = n_rows % wanted_rows

    if remaining != 0:
        remaining = wanted_rows - remaining
        last_row = image[n_rows - 1]
        rows_to_add = np.repeat([last_row], remaining, axis=0)
        image = np.vstack((image, rows_to_add))

    remaining = n_cols % wanted_cols

    if remaining != 0:
        remaining = wanted_cols - remaining
        last_col = image[:, [-1]]
        image = np.hstack((image, np.tile(last_col, (remaining, 1))))

    return image


def reverse_padding(padded_image, original_rows, original_cols):
    """
                                Reverses the padding.
                                :param padded_image: the padded image to unpad.
                                :param original_rows: number of original rows.
                                :param original_cols: number of original columns.
                                :return: the original image.
    """
    n_rows = padded_image.shape[0]
    n_cols = padded_image.shape[1]

    n_rows_to_delete = n_rows - original_rows
    if n_rows_to_delete != 0:
        rows_to_delete = np.arange(n_rows - n_rows_to_delete - 1, n_rows - 1)
        padded_image = np.delete(padded_image, rows_to_delete, axis=0)

    n_cols_to_delete = n_cols - original_cols
    if n_cols_to_delete != 0:
        cols_to_delete = np.arange(n_cols - n_cols_to_delete - 1, n_cols - 1)
        padded_image = np.delete(padded_image, cols_to_delete, axis=1)

    return padded_image