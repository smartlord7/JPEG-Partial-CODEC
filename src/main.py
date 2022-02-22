import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
import os


def show_images(images):
    for image_name in images.keys():
        plt.figure()
        plt.imshow(images[image_name])
        plt.show()


def read_images(directory, ext):
    images = dict()

    for image_name in os.listdir(directory):
        if image_name.endswith(ext):
            image = plt.imread(directory + image_name)
            print("Read %s - shape: %s, type: %s" % (image_name, image.shape, image.dtype))
            images[image_name] = image

    return images


def read_images2(directory, ext):
    images = dict()

    for image_name in os.listdir(directory):
        if image_name.endswith(ext):
            image = Image.open(directory + image_name)
            images[image_name] = image

    return images


def separate_rgb(img):
    r, g, b = img.copy(), img.copy(), img.copy()
    r[:, :, (1, 2)] = 0
    g[:, :, (0, 2)] = 0
    b[:, :, (0, 1)] = 0
    img_rgb = np.concatenate((r, g, b))
    plt.figure()
    plt.imshow(img_rgb)
    plt.show()

    return img_rgb


def float_to_uint8(matrix):
    matrix = matrix.round()
    matrix[matrix > 255] = 255
    matrix[matrix < 0] = 0
    matrix = matrix.astype(np.uint8)

    return matrix


def rgb_to_y_cb_cr(rgb, y_cb_cr_matrix):
    y_cb_cr = rgb.dot(y_cb_cr_matrix.T)
    y_cb_cr[:, :, [1, 2]] += 128
    y_cb_cr = float_to_uint8(y_cb_cr)
    plt.figure()
    plt.imshow(y_cb_cr)
    plt.show()

    return y_cb_cr


def jpeg_compress_images(directory, ext, out_dir, quality_rates):
    images = read_images2(directory, ext)
    fig, axis = plt.subplots(len(images), len(quality_rates))
    i = 0

    for image_name in images.keys():
        j = 0
        for quality_rate in quality_rates:
            compressed_image_name = image_name.replace(ext, "") + str(quality_rate) + ".jpg"
            compress_image_path = out_dir + "/" + compressed_image_name
            images[image_name].save(compress_image_path, quality=quality_rate)
            image = plt.imread(compress_image_path)
            axis[i, j].imshow(image)
            axis[i, j].set_title(compressed_image_name, fontsize=10)

            j += 1

        i += 1

    plt.show()


def generate_linear_colormap(color_list):
    colormap = clr.LinearSegmentedColormap('cmap', color_list, N=256)

    return colormap


def apply_padding(image, wanted_rows, wanted_cols):
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    padded_image = np.copy(image)

    remaining = n_rows % wanted_rows

    if remaining != 0:
        last_row = n_rows[n_rows - 1]
        rows_to_add = np.repeat([last_row], remaining, axis=0)
        padded_image = np.vstack((image, rows_to_add))

    remaining = n_cols % wanted_cols

    if n_cols % wanted_cols != 0:
        last_col = n_rows[:, : n_cols - 1]
        cols_to_add = np.repeat(last_col, axis=1)

        padded_image = np.insert(image, n_cols, last_col, axis=1)

    return padded_image


def encoder(image, params):
    pass


def decoder(encoded_image):
    pass


def main():
    CWD = os.getcwd()
    ORIGINAL_IMAGE_DIRECTORY = CWD + '/original_img/'
    ORIGINAL_IMAGE_EXTENSION = '.bmp'
    COMPRESSED_IMAGE_DIRECTORY = CWD + '\\jpeg_compressed_img'
    JPEG_QUALITY_RATES = [25, 50, 75]
    Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    Y_CB_CR_MATRIX_INVERSE = np.linalg.inv(Y_CB_CR_MATRIX)

    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    #show_images(original_images)
    #compressed_images = jpeg_compress_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION, COMPRESSED_IMAGE_DIRECTORY, JPEG_QUALITY_RATES)
    #show_images(compressed_images)
    img_rgb = separate_rgb(original_images["barn_mountains.bmp"])
    y_cb_cr = rgb_to_y_cb_cr(img_rgb, Y_CB_CR_MATRIX)

    #cmap = generate_linear_colormap([(1, 0, 0), (1, 0, 0)])


if __name__ == '__main__':
    main()
