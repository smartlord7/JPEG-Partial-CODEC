import numpy as np
from astropy.nddata import reshape_as_blocks
from numpy import r_
from scipy.fftpack import dct, idct


def split_matrix_blockwise(array, block_size):
    """
    Split a matrix into sub-matrices.
    :param array: the matrix to split
    :param block_size: the block size to reshape
    :return: the reshaped matrix.
    """

    return reshape_as_blocks(array, block_size)


def join_matrix_blockwise(blocks):
    """
    Joins submatrixes into an matrix
    :param blocks: the blocks to join in the matrix
    :return: the concatenated matrix
    """

    return np.concatenate(np.concatenate(blocks, axis=1), axis=1)


def apply_dct_blocks_optimized(im, block_size):
    """
    Applies DCT in blocks (main function)
    :param im: the image to apply the dct blocks
    :param block_size: the block size to be applied
    :return: the dct blocks.
    """
    blocks = split_matrix_blockwise(im, block_size)
    dct_blocks = dct(dct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")

    return dct_blocks


def apply_inverse_dct_blocks_optimized(blocks):
    """
    Applies inverse DCT in blocks (main function)
    :param blocks: the blocks to be applied
    :return: the image
    """
    idct_blocks = idct(idct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    image = join_matrix_blockwise(idct_blocks)

    return image


def apply_dct_blocks_r_(im, block_size):
    """
    Applies DCT in blocks (test function w/r)
    :param im: the image to apply the dct blocks
    :param block_size: the block size to be applied
    :return: the dct image.
    """
    imsize = im.shape
    dct_image = np.zeros(imsize)

    for i in r_[:imsize[0]:block_size]:
        for j in r_[:imsize[1]:block_size]:
            dct_block = apply_dct(im[i:(i + block_size), j:(j + block_size)])
            dct_image[i:(i + block_size), j:(j + block_size)] = dct_block

    return dct_image


def apply_inverse_dct_blocks_r_(dct_image, block_size):
    """
    Applies inverse DCT in blocks (test function w/r)
    :param dct_image: the dct image to apply the blocks
    :param block_size: the block size to be applied
    :return: the image.
    """
    imsize = dct_image.shape
    image = np.zeros(imsize)

    for i in r_[:imsize[0]:block_size]:
        for j in r_[:imsize[1]:block_size]:
            image[i:(i + block_size), j:(j + block_size)] = apply_inverse_dct(
                dct_image[i:(i + block_size), j:(j + block_size)])

    return image


def apply_dct_blocks_loops(im, block_size):
    """
    Applies DCT in blocks (w/loops)
    :param im: the image to apply the dct blocks
    :param block_size: the block size to be applied
    :return: the dct image.
    """
    imsize = im.shape
    dct_image = np.zeros(imsize)

    for i in range(0, imsize[0], block_size):
        for j in range(0, imsize[1], block_size):
            dct_block = apply_dct(im[i:(i + block_size), j:(j + block_size)])
            dct_image[i:(i + block_size), j:(j + block_size)] = dct_block

    return dct_image


def apply_inverse_dct_blocks_loops(dct_image, block_size):
    """
    Applies inverse DCT in blocks (w/loops)
    :param dct_image: the dct image to apply the blocks
    :param block_size: the block size to be applied
    :return: the image.
    """
    imsize = dct_image.shape
    image = np.zeros(imsize)

    for i in range(0, imsize[0], block_size):
        for j in range(0, imsize[1], block_size):
            image[i:(i + block_size), j:(j + block_size)] = apply_inverse_dct(
                dct_image[i:(i + block_size), j:(j + block_size)])

    return image


def apply_dct(matrix):
    """
    Applies DCT in the matrix
    :param matrix: the matrix where the DCT will be applied
    :return: the matrix with the DCT applied
    """
    matrix_dct = dct(dct(matrix, norm="ortho").T, norm="ortho").T

    return matrix_dct


def apply_inverse_dct(matrix_dct):
    """
    Applies inverse DCT in the matrix
    :param matrix_dct: the matrix where the IDCT will be applied
    :return: the matrix with the IDCT applied
    """
    matrix_idct = idct(idct(matrix_dct, norm="ortho").T, norm="ortho").T

    return matrix_idct


def plot_f(image):
    """
    Function to apply before plotting the data
    """
    return np.log2(np.abs(image) + 0.0001)
