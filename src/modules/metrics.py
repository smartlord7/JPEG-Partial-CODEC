import numpy as np


def calc_error_image(original_image, decompressed_image):
    """
    Function that calculates the image error:
    :param original_image: The original image
    :param decompressed_image: The decompressed image
    :return: the calculated error of the image
    """
    return np.abs(original_image - decompressed_image)


def calc_mse(original_image, decompressed_image):
    """
    Function to calculate the MSE:
    :param original_image: The original image
    :param decompressed_image: The decompressed image
    :return: the MSE of the image
    """
    n = original_image.shape[0] * original_image.shape[1]

    return np.sum((calc_error_image(original_image.astype(np.float), decompressed_image.astype(np.float)) ** 2)) / n


def calc_rmse(original_image, decompressed_image):
    """
    Function to calculate the RMSE:
    :param original_image: The original image
    :param decompressed_image: The decompressed image
    :return: the RMSE of the image
    """
    return calc_mse(original_image, decompressed_image) ** (1/2)


def calc_snr(original_image, decompressed_image):
    """
    Function to calculate the SNR:
    :param original_image: The original image
    :param decompressed_image: The decompressed image
    :return: the SNR of the image
    """

    n = original_image.shape[0] * original_image.shape[1]
    power = np.sum((original_image.astype(np.float) ** 2)) / n

    return 10 * np.log10(power / calc_mse(original_image, decompressed_image))


def calc_psnr(original_image, decompressed_image):
    """
    Function to calculate the PSNR:
    :param original_image: The original image
    :param decompressed_image: The decompressed image
    :return: the PSNR of the image
    """
    max_val = np.max(original_image.astype(np.float)) ** 2

    return 10 * np.log10(max_val / calc_mse(original_image, decompressed_image))


def show_jpeg_metrics(original_image, decompressed_image):
    """
    Function to show the calculated metrics
    :param original_image: The original image
    :param decompressed_image: The decompressed image
    """
    print("MSE: %.5f \n"
          "RMSE: %.5f \n"
          "SNR: %.5f \n"
          "PSNR: %.5f" %
          (calc_mse(original_image, decompressed_image),
          calc_rmse(original_image, decompressed_image),
          calc_snr(original_image, decompressed_image),
          calc_psnr(original_image, decompressed_image)))
