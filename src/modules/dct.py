import numpy as np
from astropy.nddata import reshape_as_blocks
from matplotlib import pyplot as plt
from numpy import r_
from scipy.fftpack import dct, idct


def split_matrix_blockwise(array, block_size):
    """Split a matrix into sub-matrices."""

    return reshape_as_blocks(array, block_size)


def join_matrix_blockwise(blocks):
    """Joins submatrixes into an matrix"""

    return np.concatenate(np.concatenate(blocks, axis=1), axis=1)


def apply_dct_blocks_optimized(im, block_size, cmap):
    blocks = split_matrix_blockwise(im, block_size)
    dct_blocks = dct(dct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    dct_image = join_matrix_blockwise(dct_blocks)

    plt.figure()
    plt.imshow(dct_image, cmap=cmap)
    plt.title(str(block_size) + "x" + str(block_size) + " DCT blocks")

    return dct_blocks


def apply_inverse_dct_blocks_optimized(blocks, cmap):
    idct_blocks = idct(idct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    image = join_matrix_blockwise(idct_blocks)

    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title("IDCT blocks")

    return image


def apply_dct_blocks(im, block_size, cmap):
    imsize = im.shape
    dct_image = np.zeros(imsize)

    for i in r_[:imsize[0]:block_size]:
        for j in r_[:imsize[1]:block_size]:
            dct_block = apply_dct(im[i:(i + block_size), j:(j + block_size)])
            dct_image[i:(i + block_size), j:(j + block_size)] = dct_block

    plt.figure()
    plt.imshow(dct_image, cmap=cmap)
    plt.title(str(block_size) + "x" + str(block_size) + "DCT blocks")

    return dct_image


def apply_inverse_dct_blocks(image_name, dct_image, block_size):
    imsize = dct_image.shape
    image = np.zeros(imsize)

    for i in r_[:imsize[0]:block_size]:
        for j in r_[:imsize[1]:block_size]:
            image[i:(i + block_size), j:(j + block_size)] = apply_inverse_dct(
                dct_image[i:(i + block_size), j:(j + block_size)])

    plt.figure()
    plt.imshow(image)
    plt.title(image_name + "-" + str(block_size) + "x" + str(block_size) + "Inverse DCT blocks")

    return image


def apply_dct(matrix):
    matrix_dct = dct(dct(matrix, norm="ortho").T, norm="ortho").T

    return matrix_dct


def apply_inverse_dct(matrix_dct):
    matrix_idct = idct(idct(matrix_dct, norm="ortho").T, norm="ortho").T

    return matrix_idct