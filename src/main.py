import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as clr


def show_images(images):
    t = type(images)

    if t == dict:
        for image_name in images:
            plt.figure()
            plt.title(image_name)
            plt.imshow(images[image_name])
            plt.show()
    elif t == list:
        for image in images:
            plt.figure()
            plt.imshow(image)
            plt.show()
    else:
        plt.figure()
        plt.imshow(images)
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

    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def join_rgb(r, g, b):
    shape = (r.shape[0], r.shape[1], 3)
    image = np.zeros(shape, np.uint8)
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b

    return image


def float_to_uint8(matrix):
    matrix = matrix.round()
    matrix[matrix > 255] = 255
    matrix[matrix < 0] = 0
    matrix = matrix.astype(np.uint8)

    return matrix


def rgb_to_y_cb_cr(rgb, y_cb_cr_matrix):
    y_cb_cr = rgb.dot(y_cb_cr_matrix.T)
    y_cb_cr[:, :, [1, 2]] += 128
    plt.figure()

    plt.imshow(np.concatenate((y_cb_cr[:, :, 0], y_cb_cr[:, :, 1], y_cb_cr[:, :, 2])))
    plt.show()

    return y_cb_cr


def y_cb_cr_to_rgb(y_cb_cr_inverse_matrix, y_cb_cr):
    y_cb_cr[:, :, [1, 2]] -= 128
    rgb = y_cb_cr.dot(y_cb_cr_inverse_matrix.T)
    rgb = float_to_uint8(rgb)
    plt.imshow(rgb)
    plt.show()

    return rgb


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
    colormap = clr.LinearSegmentedColormap.from_list('cmap', color_list, N=256)

    return colormap


def plot_image_colormap(image_channel, colormap):
    plt.figure()
    plt.imshow(image_channel, colormap)
    plt.show()


def apply_padding(image, wanted_rows, wanted_cols):
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    remaining = n_rows % wanted_rows

    if remaining != 0:
        remaining = wanted_rows - remaining
        last_row = image[n_rows - 1]
        rows_to_add = np.repeat([last_row], remaining, axis=0)
        image = np.vstack((image, rows_to_add))

    remaining = n_cols % wanted_cols

    if remaining != 0:
        remaining = wanted_cols - remaining
        last_col = image[:, [-1]]
        image = np.hstack((image, np.tile(last_col, (remaining, 1))))

    return image


def reverse_padding(padded_image, original_rows, original_cols):
    n_rows = padded_image.shape[0]
    n_cols = padded_image.shape[1]

    rows_to_delete = n_rows - original_rows
    rows_to_delete = [i for i in range(n_rows - rows_to_delete - 1, n_rows - 1)]

    if rows_to_delete != 0:
        padded_image = np.delete(padded_image, rows_to_delete, axis=0)

    cols_to_delete = n_cols - original_cols
    cols_to_delete = [i for i in range(n_cols - cols_to_delete - 1, n_cols - 1)]

    if cols_to_delete != 0:
        padded_image = np.delete(padded_image, cols_to_delete, axis=1)

    return padded_image


def encoder(image):
    Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    GREY_CMAP_LIST = [(0, 0, 0), (1, 1, 1)]
    RED_CMAP_LIST = [(0, 0, 0), (1, 0, 0)]
    GREEN_CMAP_LIST = [(0, 0, 0), (0, 1, 0)]
    BLUEE_CMAP_LIST = [(0, 0, 0), (0, 0, 1)]

    grey_cmap = generate_linear_colormap(GREY_CMAP_LIST)
    red_cmap = generate_linear_colormap(RED_CMAP_LIST)
    green_cmap = generate_linear_colormap(GREEN_CMAP_LIST)
    blue_cmap = generate_linear_colormap(BLUEE_CMAP_LIST)
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    r, g, b = separate_rgb(image)
    plot_image_colormap(r, red_cmap)
    plot_image_colormap(g, green_cmap)
    plot_image_colormap(b, blue_cmap)

    padded_image = apply_padding(image, 32, 32)
    show_images(padded_image)

    y_cb_cr_image = rgb_to_y_cb_cr(image, Y_CB_CR_MATRIX)
    show_images(y_cb_cr_image)
    plot_image_colormap(y_cb_cr_image[:, :, 0], grey_cmap)
    plot_image_colormap(y_cb_cr_image[:, :, 1], grey_cmap)
    plot_image_colormap(y_cb_cr_image[:, :, 2], grey_cmap)

    return n_rows, n_cols


def decoder(encoded_image, n_rows, n_cols):
    Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    Y_CB_CR_MATRIX_INVERSE = np.linalg.inv(Y_CB_CR_MATRIX)

    rgb_image = rgb_to_y_cb_cr(encoded_image, Y_CB_CR_MATRIX_INVERSE)
    unpadded_image = reverse_padding(rgb_image, n_rows, n_cols)


def main():
    CWD = os.getcwd()
    ORIGINAL_IMAGE_DIRECTORY = CWD + '/original_img/'
    ORIGINAL_IMAGE_EXTENSION = '.bmp'
    COMPRESSED_IMAGE_DIRECTORY = CWD + '\\jpeg_compressed_img'
    JPEG_QUALITY_RATES = [25, 50, 75]
    Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    Y_CB_CR_MATRIX_INVERSE = np.linalg.inv(Y_CB_CR_MATRIX)
    RGB_MATRIX = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    GREY_CMAP_LIST = [(0, 0, 0), (1, 1, 1)]
    RED_CMAP_LIST = [(0, 0, 0), (1, 0, 0)]
    GREEN_CMAP_LIST = [(0, 0, 0), (0, 1, 0)]
    BLUEE_CMAP_LIST = [(0, 0, 0), (0, 0, 1)]
    grey_cmap = generate_linear_colormap(GREY_CMAP_LIST)
    red_cmap = generate_linear_colormap(RED_CMAP_LIST)
    green_cmap = generate_linear_colormap(GREEN_CMAP_LIST)
    blue_cmap = generate_linear_colormap(BLUEE_CMAP_LIST)

    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    show_images(original_images)
    jpeg_compress_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION, COMPRESSED_IMAGE_DIRECTORY, JPEG_QUALITY_RATES)
    img = original_images["logo.bmp"]
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    r, g, b = separate_rgb(img)
    joined_rgb = join_rgb(r, g, b)
    show_images(joined_rgb)
    plot_image_colormap(r, red_cmap)
    plot_image_colormap(g, green_cmap)
    plot_image_colormap(b, blue_cmap)
    padded_image = apply_padding(img, 16, 16)
    show_images(padded_image)
    unpadded_image = reverse_padding(padded_image, n_rows, n_cols)
    show_images(unpadded_image)

    y_cb_cr = rgb_to_y_cb_cr(original_images["peppers.bmp"], Y_CB_CR_MATRIX)
    plot_image_colormap(y_cb_cr[:, :, 0], grey_cmap)
    plot_image_colormap(y_cb_cr[:, :, 1], blue_cmap)
    plot_image_colormap(y_cb_cr[:, :, 2], red_cmap)
    rgb = y_cb_cr_to_rgb(Y_CB_CR_MATRIX_INVERSE, y_cb_cr)


if __name__ == '__main__':
    main()