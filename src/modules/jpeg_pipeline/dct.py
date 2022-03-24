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
from astropy.nddata import reshape_as_blocks
from numpy import r_
from scipy.fftpack import dct, idct


def split_matrix_blockwise(array: np.ndarray, block_size: int):
    """
    Splits a matrix into sub-matrices.
    :param array: the matrix to split
    :param block_size: the block size to reshape
    :return: the split matrix in blocks of block_size size.
    """

    return reshape_as_blocks(array, block_size)


def join_matrix_blockwise(blocks: np.ndarray):
    """
    Joins sub-matrices into a matrix
    :param blocks: the sub-matrices to join
    :return: the joined matrix
    """

    return np.concatenate(np.concatenate(blocks, axis=1), axis=1)


def apply_dct_blocks_optimized(im: np.ndarray, block_size: int):
    """
    Applies DCT in blocks using an Astropy function
    :param im: the image to which the DCT in blocks will be applied
    :param block_size: the size of each block that will be submitted to the DCT
    :return: the dct blocks.
    """
    blocks = split_matrix_blockwise(im, block_size)
    dct_blocks = dct(dct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")

    return dct_blocks


def apply_inverse_dct_blocks_optimized(blocks: np.ndarray):
    """
    Applies inverse DCT in blocks
    :param blocks: the image to which the DCT in blocks will be applied
    :return: the IDCT blocks
    """
    idct_blocks = idct(idct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    image = join_matrix_blockwise(idct_blocks)

    return image


def apply_dct_blocks_r_(im: np.ndarray, block_size: int):
    """
    Applies DCT in blocks (test function w/r_)
    :param im: the image to which the DCT in blocks will be applied
    :param block_size: the size of each block that will be submitted to the DCT
    :return: the DCT image.
    """
    imsize = im.shape
    dct_image = np.zeros(imsize)

    for i in r_[:imsize[0]:block_size]:
        for j in r_[:imsize[1]:block_size]:
            dct_block = apply_dct(im[i:(i + block_size), j:(j + block_size)])
            dct_image[i:(i + block_size), j:(j + block_size)] = dct_block

    return dct_image


def apply_inverse_dct_blocks_r_(dct_image: np.ndarray, block_size: int):
    """
    Applies inverse DCT in blocks (test function w/r)
    :param dct_image: the blocks to which the IDCT will be applied
    :param block_size: the size of each block that will be submitted to the IDCT
    :return: the original image.
    """
    imsize = dct_image.shape
    image = np.zeros(imsize)

    for i in r_[:imsize[0]:block_size]:
        for j in r_[:imsize[1]:block_size]:
            image[i:(i + block_size), j:(j + block_size)] = apply_inverse_dct(
                dct_image[i:(i + block_size), j:(j + block_size)])

    return image


def apply_dct_blocks_loops(im: np.ndarray, block_size: int):
    """
    Applies DCT in blocks (w/ raw loops)
    :param im: the blocks to which the DCT will be applied
    :param block_size: the size of each block that will be submitted to the DCT
    :return: the DCT image.
    """
    imsize = im.shape
    dct_image = np.zeros(imsize)

    for i in range(0, imsize[0], block_size):
        for j in range(0, imsize[1], block_size):
            dct_block = apply_dct(im[i:(i + block_size), j:(j + block_size)])
            dct_image[i:(i + block_size), j:(j + block_size)] = dct_block

    return dct_image


def apply_inverse_dct_blocks_loops(dct_image: np.ndarray, block_size: int):
    """
    Applies inverse DCT in blocks (w/loops)
    :param dct_image: the image to which the IDCT in blocks will be applied
    :param block_size: the size of each block that will be submitted to the DCT
    :return: the original image.
    """
    imsize = dct_image.shape
    image = np.zeros(imsize)

    for i in range(0, imsize[0], block_size):
        for j in range(0, imsize[1], block_size):
            image[i:(i + block_size), j:(j + block_size)] = apply_inverse_dct(
                dct_image[i:(i + block_size), j:(j + block_size)])

    return image


def apply_dct(matrix: np.ndarray):
    """
    Applies DCT to a matrix
    :param matrix: the matrix to which the DCT will be applied
    :return: the matrix with the DCT applied
    """
    matrix_dct = dct(dct(matrix, norm="ortho").T, norm="ortho").T

    return matrix_dct


def apply_inverse_dct(matrix_dct: np.ndarray):
    """
    Applies inverse DCT in the matrix
    :param matrix_dct: the matrix to which the IDCT  will be applied
    :return: the matrix with the IDCT applied
    """
    matrix_idct = idct(idct(matrix_dct, norm="ortho").T, norm="ortho").T

    return matrix_idct


def plot_f(image: np.ndarray):
    """
    Function to apply before plotting the data
    """
    return np.log2(np.abs(image) + 0.0001)
