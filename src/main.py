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


def separate_rgb(img, show_plots=False):
    r, g, b = img.copy(), img.copy(), img.copy()
    r[:, :, (1, 2)] = 0
    g[:, :, (0, 2)] = 0
    b[:, :, (0, 1)] = 0

    if show_plots:
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


def rgb_to_y_cb_cr(rgb, y_cb_cr_matrix, show_plots=False):
    y_cb_cr = rgb.dot(y_cb_cr_matrix.T)
    y_cb_cr[:, :, [1, 2]] += 128

    if show_plots:
        plt.figure()
        plt.imshow(np.concatenate((y_cb_cr[:, :, 0], y_cb_cr[:, :, 1], y_cb_cr[:, :, 2])))
        plt.show()

    return y_cb_cr


def y_cb_cr_to_rgb(y_cb_cr, y_cb_cr_inverse_matrix, show_plots=False):
    y_cb_cr[:, :, [1, 2]] -= 128
    rgb = y_cb_cr.dot(y_cb_cr_inverse_matrix.T)
    rgb = float_to_uint8(rgb)

    if show_plots:
        plt.figure()
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


def down_sample(cb, cr, variant, f):
    if variant == 1:
        cb_down_sampled = cb[:, 0::f]
        cr_down_sampled = cr[:, 0::f]
    elif variant == 2:
        cb_down_sampled = cb[0::f, 0::f]
        cr_down_sampled = cr[0::f, 0::f]
    else:
        return cb, cr

    return cb_down_sampled, cr_down_sampled


def up_sample(cb, cr, cb_factor, cr_factor):
    cb_t = list(zip(*cb))
    cr_t = list(zip(*cr))

    if cb_factor == 2:
        for i in range(0, len(cr_t[0]), 2):
            j = i * 2
            copy_cr_t = cr_t[j]
            cr_t.insert(j + 1, copy_cr_t)

        for i in range(0, len(cb_t[0]), 2):
            j = i * 2
            copy_cb_t = cb_t[j]
            cb.insert(j + 1, copy_cb_t)

    cb = list(zip(*cb_t))
    cr = list(zip(*cr_t))

    if cr_factor == 0:
        for i in range(len(cr[0])):
            j = i * 2
            copy_cr = cr[j]
            cr.insert(j + 1, copy_cr)

        for i in range(0, len(cb[0]), 2):
            j = i * 2
            copy_cb = cb[j]
            cb.insert(j + 1, copy_cb)

    return cb, cr


def encoder(image_data, show_plots=False):
    image_name = image_data[0]
    image_matrix = image_data[1]

    Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    GREY_CMAP_LIST = [(0, 0, 0), (1, 1, 1)]
    RED_CMAP_LIST = [(0, 0, 0), (1, 0, 0)]
    GREEN_CMAP_LIST = [(0, 0, 0), (0, 1, 0)]
    BLUE_CMAP_LIST = [(0, 0, 0), (0, 0, 1)]

    grey_cmap = generate_linear_colormap(GREY_CMAP_LIST)
    red_cmap = generate_linear_colormap(RED_CMAP_LIST)
    green_cmap = generate_linear_colormap(GREEN_CMAP_LIST)
    blue_cmap = generate_linear_colormap(BLUE_CMAP_LIST)
    n_rows = image_matrix.shape[0]
    n_cols = image_matrix.shape[1]

    padded_image = apply_padding(image_matrix, 32, 32)
    if show_plots:
        show_images(padded_image)

    r, g, b = separate_rgb(padded_image, show_plots)
    if show_plots:
        plot_image_colormap(r, red_cmap)
        plot_image_colormap(g, green_cmap)
        plot_image_colormap(b, blue_cmap)

    y_cb_cr_image = rgb_to_y_cb_cr(padded_image, Y_CB_CR_MATRIX, show_plots)
    y_cb_cr_image_as_uint8 = float_to_uint8(y_cb_cr_image)

    y = y_cb_cr_image_as_uint8[:, :, 0]
    cb = y_cb_cr_image_as_uint8[:, :, 1]
    cr = y_cb_cr_image_as_uint8[:, :, 2]

    if show_plots:
        plot_image_colormap(y, grey_cmap)
        plot_image_colormap(cb, grey_cmap)
        plot_image_colormap(cr, grey_cmap)

    down_sampled_image = down_sample(cb, cr, 1, 2)

    return down_sampled_image, n_rows, n_cols


def decoder(encoded_image_data):
    encoded_image_name = encoded_image_data[0]
    encoded_image = encoded_image_data[1]
    original_rows = encoded_image_data[2]
    original_cols = encoded_image_data[3]

    Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    Y_CB_CR_MATRIX_INVERSE = np.linalg.inv(Y_CB_CR_MATRIX)

    rgb_image = y_cb_cr_to_rgb(encoded_image, Y_CB_CR_MATRIX_INVERSE)
    unpadded_image = reverse_padding(rgb_image, original_rows, original_cols)
    show_images(unpadded_image)

    decoded_image = unpadded_image

    return decoded_image


def image_equals(original_image, decoded_image):
    return np.allclose(original_image, decoded_image)


def main():
    CWD = os.getcwd()
    ORIGINAL_IMAGE_DIRECTORY = CWD + '/original_img/'
    ORIGINAL_IMAGE_EXTENSION = '.bmp'
    COMPRESSED_IMAGE_DIRECTORY = CWD + '\\jpeg_compressed_img'
    JPEG_QUALITY_RATES = [25, 50, 75]

    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    show_images(original_images)
    jpeg_compress_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION, COMPRESSED_IMAGE_DIRECTORY, JPEG_QUALITY_RATES)

    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder((image_name, original_images[image_name]), False)
        encoded_images[image_name] = (result[0], result[1], result[2])

    decoded_images = dict()
    for encoded_image_name in encoded_images.keys():
        data = encoded_images[encoded_image_name]
        result = decoder((encoded_image_name, data[0], data[1], data[2]))
        decoded_images[encoded_image_name] = result

        if image_equals(original_images[encoded_image_name], result):
            print("Compression successful")
        else:
            print("Compression unsuccessful")


if __name__ == '__main__':
    main()