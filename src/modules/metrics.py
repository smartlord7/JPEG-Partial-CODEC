import numpy as np


def calc_error_image(original_image, decompressed_image):
    return np.abs(original_image - decompressed_image)


def calc_mse(original_image, decompressed_image):
    return np.mean(calc_error_image(original_image, decompressed_image) ** 2)


def calc_rmse(original_image, decompressed_image):
    return calc_mse(original_image, decompressed_image) ** (1/2)


def calc_snr(original_image, decompressed_image):
    power = np.mean(original_image ** 2)

    return 10 * np.log10(power / calc_mse(original_image, decompressed_image))


def calc_psnr(original_image, decompressed_image):
    max_val = np.max(original_image)

    return 10 * np.log10(max_val / calc_mse(original_image, decompressed_image))


def show_jpeg_metrics(original_image, decompressed_image):
    print("MSE: %.5f \n"
          "RMSE: %.5f \n"
          "SNR: %.5f \n"
          "PSNR: %.5f" %
          (calc_mse(original_image, decompressed_image),
          calc_rmse(original_image, decompressed_image),
          calc_snr(original_image, decompressed_image),
          calc_psnr(original_image, decompressed_image)))
