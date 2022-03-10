"""------------COMPRESSAO DE IMAGEM------------
Universidade de Coimbra
Licenciatura em Engenharia Informatica
Multimedia
Terceiro ano, segundo semestre
Authors:
Rui Bernardo Lopes Rodrigues
Sancho Amaral Simões, 2019217590, uc2019217590@student.uc.pt
Tiago Filipe Santa Ventura, 2019243695, uc2019243695@student.uc.pt
19/12/2020
---------------------------------------------------------------------------"""

from modules.dct import *
from modules.image import *
from modules.jpeg_pipeline.padding import *
from modules.jpeg_pipeline.sampling import *
from modules.jpeg_pipeline.y_cb_cr import *
from modules.util import *


def view_dct(y, cb, cr, cmap, name):
    plt.figure()
    plt.title(name + " Y")
    plt.imshow(np.log2(np.abs(y) + 0.0001), cmap)
    plt.show()
    plt.figure()
    plt.title(name + " Cb")
    plt.imshow(np.log2(np.abs(cb) + 0.0001), cmap)
    plt.show()
    plt.figure()
    plt.title(name + " Cr")
    plt.imshow(np.log2(np.abs(cr) + 0.0001), cmap)
    plt.show()


def encoder(image_data, show_plots=False):
    """
                                       Enconder function.
                                       :param image_data: the image to encode.
                                       :param show_plots: flag that enables plotting.
                                       :return: the encoded image.
    """
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
        show_images(padded_image,"Padded Image")

    r, g, b = separate_rgb(padded_image, show_plots)
    if show_plots:
        show_images(r, red_cmap, "Red")
        show_images(g, green_cmap, "Green")
        show_images(b, blue_cmap, "Blue")

    y_cb_cr_image = rgb_to_y_cb_cr(padded_image, Y_CB_CR_MATRIX, show_plots)

    y = y_cb_cr_image[:, :, 0]
    cb = y_cb_cr_image[:, :, 1]
    cr = y_cb_cr_image[:, :, 2]

    if show_plots:
        show_images(y, grey_cmap, "Y")
        show_images(cb, grey_cmap, "Cb")
        show_images(cr, grey_cmap, "Cr")

    cb, cr = down_sample(cb, cr, 1, 2)

    #y_dct_total = apply_dct(y)
    #cb_dct_total = apply_dct(cb)
    #cr_dct_total = apply_dct(cr)

    #view_dct(y_dct_total, cb_dct_total, cr_dct_total, grey_cmap, "DCT")

    #y_idct = apply_inverse_dct(y_dct_total)
    #cb_idct = apply_inverse_dct(cb_dct_total)
    #cr_idct = apply_inverse_dct(cr_dct_total)

    #view_dct(y_idct, cb_idct, cr_idct, grey_cmap, "IDCT")

    y_dct_blocks = apply_dct_blocks_optimized(y, 16, grey_cmap)
    cb_dct_blocks = apply_dct_blocks_optimized(cb, 16, grey_cmap)
    cr_dct_blocks = apply_dct_blocks_optimized(cr, 16, grey_cmap)

    return (y_dct_blocks, cb_dct_blocks, cr_dct_blocks), n_rows, n_cols


def decoder(encoded_image_data):
    """
                                           Decode function.
                                           :param encoded_image_data: the image to decode.
                                           :return: the decoded image.
    """
    encoded_image_name = encoded_image_data[0]
    encoded_image = encoded_image_data[1]
    original_rows = encoded_image_data[2]
    original_cols = encoded_image_data[3]

    Y_CB_CR_MATRIX = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    Y_CB_CR_MATRIX_INVERSE = np.linalg.inv(Y_CB_CR_MATRIX)
    GREY_CMAP_LIST = [(0, 0, 0), (1, 1, 1)]
    grey_cmap = generate_linear_colormap(GREY_CMAP_LIST)

    y = encoded_image[0]
    cb = encoded_image[1]
    cr = encoded_image[2]
    y_inverse_dct = apply_inverse_dct_blocks_optimized(y, grey_cmap)
    cb_inverse_dct = apply_inverse_dct_blocks_optimized(cb, grey_cmap)
    cr_inverse_dct = apply_inverse_dct_blocks_optimized(cr, grey_cmap)
    cb_up_sampled, cr_up_sampled = up_sample(cb_inverse_dct, cr_inverse_dct, 1, 2)
    joined_channels_img = join_channels(y_inverse_dct, cb_up_sampled, cr_up_sampled)
    rgb_image = y_cb_cr_to_rgb(joined_channels_img, Y_CB_CR_MATRIX_INVERSE)
    unpadded_image = reverse_padding(rgb_image, original_rows, original_cols)
    show_images(unpadded_image, encoded_image_name + " - Decompressed")

    decoded_image = unpadded_image

    return decoded_image


def image_equals(original_image, decoded_image):
    """
                                           Verifies if the images are equal.
                                           :param original_image: original image.
                                           :param decoded_image: decoded image.
                                           :return: if the image is equal or no.
    """
    return np.allclose(original_image, decoded_image)


def main():
    """
    Main function
    """
    CWD = os.getcwd()
    ORIGINAL_IMAGE_DIRECTORY = CWD + '\\resources\\img\\original_img\\'
    ORIGINAL_IMAGE_EXTENSION = '.bmp'
    COMPRESSED_IMAGE_DIRECTORY = CWD + '\\jpeg_compressed_img'
    JPEG_QUALITY_RATES = [25, 50, 75]

    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    #show_images(original_images)
    #jpeg_compress_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION, COMPRESSED_IMAGE_DIRECTORY, JPEG_QUALITY_RATES)

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
            print("No diff")
        else:
            print("Diff")


if __name__ == '__main__':
    main()