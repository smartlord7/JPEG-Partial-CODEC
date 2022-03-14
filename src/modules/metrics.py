import numpy as np


def calc_error_image(original_image, decompressed_image):
    return np.abs(original_image - decompressed_image)


def calc_mse(original_image, compressed_image):
    pass


def calc_rmse(original_image, compressed_image):
    pass


def calc_snr(original_image, compressed_image):
    pass


def calc_psnr(original_image, compressed_image):
    pass
