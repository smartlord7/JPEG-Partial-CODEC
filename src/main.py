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
    y_cb_cr = float_to_uint8(y_cb_cr)
    plt.figure()

    plt.imshow(np.concatenate((y_cb_cr[:, :, 0], y_cb_cr[:, :, 1], y_cb_cr[:, :, 2])))
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
    colormap = clr.LinearSegmentedColormap.from_list('cmap', color_list, N=256)

    return colormap


def plot_image_colormap(image_channel, colormap):
    plt.figure()
    plt.imshow(image_channel, colormap)
    plt.show()


def apply_padding(image, wanted_rows, wanted_cols):
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    padded_image = np.copy(image)

    remaining = n_rows % wanted_rows

    if remaining != 0:
        remaining = wanted_rows - remaining
        last_row = image[n_rows - 1]
        rows_to_add = np.repeat([last_row], remaining, axis=0)
        padded_image = np.vstack((padded_image, rows_to_add))

    remaining = n_cols % wanted_cols

    if remaining != 0:
        remaining = wanted_cols - remaining
        last_col = padded_image[:, [-1]]
        padded_image = np.hstack((padded_image, np.tile(last_col, (remaining, 1))))

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
    GREY_CMAP_LIST = [(0, 0, 0), (0.5, 0.5, 0.5)]
    RED_CMAP_LIST = [(0, 0, 0), (1, 0, 0)]
    GREEN_CMAP_LIST = [(0, 0, 0), (0, 1, 0)]
    BLUEE_CMAP_LIST = [(0, 0, 0), (0, 0, 1)]

    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    #show_images(original_images)
    #compressed_images = jpeg_compress_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION, COMPRESSED_IMAGE_DIRECTORY, JPEG_QUALITY_RATES)
    #show_images(compressed_images)
    img = original_images["logo.bmp"]
    #r, g, b = separate_rgb(img)
    #joined_rgb = join_rgb(r, g, b)
    #show_images(joined_rgb)
    padded_image = apply_padding(img, 16, 16)
    show_images(padded_image)

    #grey_cmap = generate_linear_colormap(GREY_CMAP_LIST)
    #red_cmap = generate_linear_colormap(RED_CMAP_LIST)
    #green_cmap = generate_linear_colormap(GREEN_CMAP_LIST)
    #blue_cmap = generate_linear_colormap(BLUEE_CMAP_LIST)
    #plot_image_colormap(r, red_cmap)
    #plot_image_colormap(g, green_cmap)
    #plot_image_colormap(b, blue_cmap)

    #y_cb_cr = rgb_to_y_cb_cr(original_images["barn_mountains.bmp"], Y_CB_CR_MATRIX)


if __name__ == '__main__':
    main()
