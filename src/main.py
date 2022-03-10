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

from modules.jpeg import *
from modules.util import *
from modules.image import *
from modules.const import *
from modules.jpeg_pipeline.dct import *
from modules.jpeg_pipeline.padding import *
from modules.jpeg_pipeline.sampling import *
from modules.jpeg_pipeline.y_cb_cr import *


def encoder(image_data, show_plots=False):
    """
                                       Enconder function.
                                       :param image_data: the image to encode.
                                       :param show_plots: flag that enables plotting.
                                       :return: the encoded image.
    """
    image_name = image_data[0]
    image_matrix = image_data[1]
    n_rows = image_matrix.shape[0]
    n_cols = image_matrix.shape[1]

    padded_image = apply_padding(image_matrix, IMAGE_SIZE_DIVISOR, 32)
    new_shape = padded_image.shape
    added_rows = str(new_shape[0] - n_rows)
    added_cols = str(new_shape[1] - n_cols)

    if show_plots:
        show_images(padded_image, image_name + " - Padded - +" + added_rows + "|+" + added_cols)

    r, g, b = separate_rgb(padded_image)
    if show_plots:
        show_images(r, image_name + " - Red channel w/red cmap", RED_CMAP)
        show_images(g, image_name + " - Green channel w/green cmap", GREEN_CMAP)
        show_images(b, image_name + " - Blue channel w/blue cmap", BLUE_CMAP)

    y_cb_cr_image = rgb_to_y_cb_cr(padded_image, Y_CB_CR_MATRIX)

    y = y_cb_cr_image[:, :, 0]
    cb = y_cb_cr_image[:, :, 1]
    cr = y_cb_cr_image[:, :, 2]

    if show_plots:
        show_images(y, image_name + " - Y channel w/grey cmap", GREY_CMAP)
        show_images(cb, image_name + " - Cb channel w/grey cmap", GREY_CMAP)
        show_images(cr, image_name+ "Cr channel w/grey cmap", GREY_CMAP)

    cb, cr = down_sample(cb, cr, 1, 2)

    y_dct_total = apply_dct(y)
    cb_dct_total = apply_dct(cb)
    cr_dct_total = apply_dct(cr)

    show_images(np.log2(np.abs(y_dct_total) + 0.0001), image_name + " - Total DCT - Y", GREY_CMAP)
    show_images(np.log2(np.abs(cb_dct_total) + 0.0001), image_name + " - Total DCT - Cb", GREY_CMAP)
    show_images(np.log2(np.abs(cr_dct_total) + 0.0001), image_name + " - Total DCT - Cr", GREY_CMAP)

    y_idct_total = apply_inverse_dct(y_dct_total)
    cb_idct_total = apply_inverse_dct(cb_dct_total)
    cr_idct_total = apply_inverse_dct(cr_dct_total)

    show_images(y_idct_total, image_name + " - Total Inverse DCT - Y", GREY_CMAP)
    show_images(cb_idct_total, image_name + " - Total Inverse DCT - Cb", GREY_CMAP)
    show_images(cr_idct_total, image_name + " - Total Inverse DCT - Cr", GREY_CMAP)

    y_dct_blocks = apply_dct_blocks_optimized(y, BLOCK_SIZE, GREY_CMAP)
    cb_dct_blocks = apply_dct_blocks_optimized(cb, BLOCK_SIZE, GREY_CMAP)
    cr_dct_blocks = apply_dct_blocks_optimized(cr, BLOCK_SIZE, GREY_CMAP)

    joined_y_dct_blocks = join_matrix_blockwise(y_dct_blocks)
    joined_cb_dct_blocks = join_matrix_blockwise(cb_dct_blocks)
    joined_cr_dct_blocks = join_matrix_blockwise(cr_dct_blocks)

    title_blocks_dct = image_name + " - DCT by blocks " + str(BLOCK_SIZE) + "x" + str(BLOCK_SIZE)
    show_images(joined_y_dct_blocks, title_blocks_dct + " - Y", GREY_CMAP)
    show_images(joined_cb_dct_blocks, title_blocks_dct + " - Cb", GREY_CMAP)
    show_images(joined_cr_dct_blocks, title_blocks_dct + " - Cr", GREY_CMAP)

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
    cwd = os.getcwd()
    orig_img_dir = cwd + ORIGINAL_IMAGE_DIRECTORY
    comp_img_dir = cwd + COMPRESSED_IMAGE_DIRECTORY

    original_images = read_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION)
    show_images(original_images)
    jpeg_compress_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION, comp_img_dir, JPEG_QUALITY_RATES)

    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder((image_name, original_images[image_name]), True)
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