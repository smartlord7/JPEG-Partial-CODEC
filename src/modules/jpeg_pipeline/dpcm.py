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


def apply_dpcm_encoding(blocks):
    """
    Function that applies the DCPM encoding
    :param blocks: number of blocks to be applied
    :return: the blocks with DPCM
    """
    dc_coefficients = np.ravel(blocks[:, :, 0, 0])
    dc_coefficients = np.concatenate(([dc_coefficients[0]], np.diff(dc_coefficients)))
    dc_coefficients = dc_coefficients.reshape((blocks.shape[0], blocks.shape[1]))
    blocks[:, :, 0, 0] = dc_coefficients

    return blocks


def apply_dpcm_decoding(blocks):
    """
    Function that applies the DCPM decoding
    :param blocks: number of blocks to be applied
    :return: the blocks without DPCM
    """
    dc_coefficients = np.ravel(blocks[:, :, 0, 0])
    dc_coefficients = np.cumsum(dc_coefficients)
    dc_coefficients = dc_coefficients.reshape((blocks.shape[0], blocks.shape[1]))
    blocks[:, :, 0, 0] = dc_coefficients

    return blocks
