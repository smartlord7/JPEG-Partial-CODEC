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


# region Public Methods

def calc_error_matrix(matrix: np.ndarray, other_matrix: np.ndarray):
    """
    Function that calculates the matrix error.
    :param matrix: The original matrix.
    :param other_matrix: The other matrix.
    :return: the calculated error of the matrix.
    """
    return np.abs(matrix - other_matrix)


def calc_mse(matrix: np.ndarray, other_matrix: np.ndarray):
    """
    Function that calculate the Mean Square Deviation.
    :param matrix: The original matrix.
    :param other_matrix: The other matrix.
    :return: the MSE of the matrix.
    """
    n = matrix.shape[0] * matrix.shape[1]

    return np.sum((calc_error_matrix(matrix.astype(np.float), other_matrix.astype(np.float)) ** 2)) / n


def calc_rmse(matrix: np.ndarray, other_matrix: np.ndarray):
    """
    Function that calculates the Root Mean Square Deviation.
    :param matrix: the original matrix.
    :param other_matrix: the other matrix.
    :return: the RMSE of the matrix.
    """
    return calc_mse(matrix, other_matrix) ** (1 / 2)


def calc_snr(matrix: np.ndarray, other_matrix: np.ndarray):
    """
    Function that calculate the Signal to Noise Ration.
    :param matrix: The original matrix.
    :param other_matrix: The other matrix.
    :return: the SNR of the matrix.
    """

    n = matrix.shape[0] * matrix.shape[1]
    power = np.sum((matrix.astype(np.float) ** 2)) / n

    return 10 * np.log10(power / calc_mse(matrix, other_matrix))


def calc_psnr(matrix: np.ndarray, other_matrix: np.ndarray):
    """
    Function to calculate the PSNR:
    :param matrix: The original matrix
    :param other_matrix: The other matrix
    :return: the PSNR of the matrix
    """
    max_val = np.max(matrix.astype(np.float)) ** 2

    return 10 * np.log10(max_val / calc_mse(matrix, other_matrix))


def show_jpeg_metrics(matrix: np.ndarray, other_matrix: np.ndarray, output_file):
    """
    Function to show the calculated metrics
    :param matrix: The original matrix
    :param other_matrix: The other matrix
    """
    out(output_file, "Distortion metrics\n"
                     "MSE: %.5f \n"
                     "RMSE: %.5f \n"
                     "SNR: %.5f \n"
                     "PSNR: %.5f" %
        (calc_mse(matrix, other_matrix),
         calc_rmse(matrix, other_matrix),
         calc_snr(matrix, other_matrix),
         calc_psnr(matrix, other_matrix)))


def calc_entropic_stats(name, arrays, channels, info, output_file, directory=os.getcwd()):
    i = 0

    out(output_file, "................................................")
    f = plt.figure(figsize=(19.8, 10.8))
    for array in arrays:
        l = array.shape[0] * array.shape[1]
        array = np.ndarray.flatten(array)
        ax = f.add_subplot(1, 3, i + 1)
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
    f.savefig(directory + "\\" + name + info + "entropic.png", dpi=100)
    f.clear()
    plt.close(f)
    plt.cla()
    plt.clf()

    del f

# endregion Public Methods
