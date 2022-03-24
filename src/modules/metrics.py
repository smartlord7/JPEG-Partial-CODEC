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

from modules.const import *
from modules.entropy import *
from modules.util import out


# region Public Functions

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


def show_jpeg_metrics(original_image, decompressed_image, output_file):
    """
    Function to show the calculated metrics
    :param original_image: The original image
    :param decompressed_image: The decompressed image
    """
    out(output_file, "Distortion metrics\n"
          "MSE: %.5f \n"
          "RMSE: %.5f \n"
          "SNR: %.5f \n"
          "PSNR: %.5f" %
          (calc_mse(original_image, decompressed_image),
          calc_rmse(original_image, decompressed_image),
          calc_snr(original_image, decompressed_image),
          calc_psnr(original_image, decompressed_image)))

    """
    Function to show the calculated entropic stats
    :param name: Name of the image/file
    :param arrays: list of arrays
    :param channels: the used channels
    :param info: the image information
    :param output_file: the output file with the information
    :param directory: the directory to put the histogram images
    """
def calc_entropic_stats(name, arrays, channels, info, output_file, directory=os.getcwd()):
    i = 0

    out(output_file, "................................................")
    fig = plt.figure()
    for array in arrays:
        l = array.shape[0] * array.shape[1]
        array = np.ndarray.flatten(array)
        ax = fig.add_subplot(1, 3, i + 1)
        hist = gen_histogram(array, 256)
        entropy_val = entropy(hist, l)
        out(output_file, "Entropy %s - %s: %.2f bits" % (channels[i], info, entropy_val))
        plt.xticks(fontsize=5)
        plt.title("%s %s Histogram" % (name, channels[i]))
        plt.ylabel('Number of occurrences')
        plt.xlabel('Symbol')
        ax.bar(ALPHABET, hist)
        i += 1
    out(output_file, "................................................")

    fig.savefig(directory + "\\" + name + info + "entropic.png")

# endregion Public Functions
