import numpy as np


def apply_dpcm_encoding(blocks):
    dc_coefficients = np.ravel(blocks[:, :, 0, 0])
    dc_coefficients = np.concatenate(([dc_coefficients[0]], np.diff(dc_coefficients)))
    dc_coefficients = dc_coefficients.reshape((blocks.shape[0], blocks.shape[1]))
    blocks[:, :, 0, 0] = dc_coefficients

    return blocks


def apply_dpcm_decoding():
    pass