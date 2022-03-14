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
from modules.jpeg_pipeline.y_cb_cr import *
from modules.jpeg_pipeline.sampling import *
from modules.jpeg_pipeline.quantization import *


def encoder(image_data, down_sampling_variant, down_sampling_step, block_size, quality_factor, show_plots=False):
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

    padded_image = apply_padding(image_matrix, down_sampling_step * block_size, down_sampling_step * block_size)
    new_shape = padded_image.shape
    added_rows = str(new_shape[0] - n_rows)
    added_cols = str(new_shape[1] - n_cols)

    if show_plots:
        show_images(padded_image, image_name + " - Padded - +" + added_rows + "|+" + added_cols, None, None)

    r, g, b = separate_channels(padded_image)
    if show_plots:
        show_images(r, image_name + " - Red channel w/red cmap", RED_CMAP, None)
        show_images(g, image_name + " - Green channel w/green cmap", GREEN_CMAP, None)
        show_images(b, image_name + " - Blue channel w/blue cmap", BLUE_CMAP, None)

    y_cb_cr_image = rgb_to_y_cb_cr(padded_image, Y_CB_CR_MATRIX)

    y = y_cb_cr_image[:, :, 0]
    cb = y_cb_cr_image[:, :, 1]
    cr = y_cb_cr_image[:, :, 2]

    if show_plots:
        show_images(y, image_name + " - Y channel w/grey cmap", GREY_CMAP, None)
        show_images(cb, image_name + " - Cb channel w/grey cmap", GREY_CMAP, None)
        show_images(cr, image_name+ " - Cr channel w/grey cmap", GREY_CMAP, None)

    cb, cr = down_sample(cb, cr, down_sampling_variant, down_sampling_step)

    y_dct_total = apply_dct(y)
    cb_dct_total = apply_dct(cb)
    cr_dct_total = apply_dct(cr)

    if show_plots:
        show_images(y_dct_total, image_name + " - Total DCT - Y", GREY_CMAP, plot_f)
        show_images(cb_dct_total, image_name + " - Total DCT - Cb", GREY_CMAP, plot_f)
        show_images(cr_dct_total, image_name + " - Total DCT - Cr", GREY_CMAP, plot_f)

    y_idct_total = apply_inverse_dct(y_dct_total)
    cb_idct_total = apply_inverse_dct(cb_dct_total)
    cr_idct_total = apply_inverse_dct(cr_dct_total)

    if show_plots:
        show_images(y_idct_total, image_name + " - Total Inverse DCT - Y", GREY_CMAP, None)
        show_images(cb_idct_total, image_name + " - Total Inverse DCT - Cb", GREY_CMAP, None)
        show_images(cr_idct_total, image_name + " - Total Inverse DCT - Cr", GREY_CMAP, None)

    y_dct_blocks = apply_dct_blocks_optimized(y, block_size)
    cb_dct_blocks = apply_dct_blocks_optimized(cb, block_size)
    cr_dct_blocks = apply_dct_blocks_optimized(cr, block_size)

    joined_y_dct_blocks = join_matrix_blockwise(y_dct_blocks)
    joined_cb_dct_blocks = join_matrix_blockwise(cb_dct_blocks)
    joined_cr_dct_blocks = join_matrix_blockwise(cr_dct_blocks)

    if show_plots:
        title_blocks_dct = image_name + " - DCT by blocks " + str(block_size) + "x" + str(block_size)
        show_images(joined_y_dct_blocks, title_blocks_dct + " - Y", GREY_CMAP, plot_f)
        show_images(joined_cb_dct_blocks, title_blocks_dct + " - Cb", GREY_CMAP, plot_f)
        show_images(joined_cr_dct_blocks, title_blocks_dct + " - Cr", GREY_CMAP, plot_f)

    y_blocks_quantized = apply_quantization(y_dct_blocks, quality_factor, JPEG_QUANTIZATION_Y)
    cb_blocks_quantized = apply_quantization(cb_dct_blocks, quality_factor, JPEG_QUANTIZATION_CB_CR)
    cr_blocks_quantized = apply_quantization(cr_dct_blocks, quality_factor, JPEG_QUANTIZATION_CB_CR)

    return (y_blocks_quantized, cb_blocks_quantized, cr_blocks_quantized), n_rows, \
           n_cols, down_sampling_variant, down_sampling_step, block_size, quality_factor


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
    down_sampling_variant = encoded_image_data[4]
    down_sampling_step = encoded_image_data[5]
    block_size = encoded_image_data[6]
    quality_factor = encoded_image_data[7]

    y = encoded_image[0]
    cb = encoded_image[1]
    cr = encoded_image[2]
    y_dequantized = apply_inverse_quantization(y, quality_factor, JPEG_QUANTIZATION_Y)
    cb_dequantized = apply_inverse_quantization(cb, quality_factor, JPEG_QUANTIZATION_CB_CR)
    cr_dequantized = apply_inverse_quantization(cr, quality_factor, JPEG_QUANTIZATION_CB_CR)
    y_inverse_dct = apply_inverse_dct_blocks_optimized(y_dequantized)
    cb_inverse_dct = apply_inverse_dct_blocks_optimized(cb_dequantized)
    cr_inverse_dct = apply_inverse_dct_blocks_optimized(cr_dequantized)
    cb_up_sampled, cr_up_sampled = up_sample(cb_inverse_dct, cr_inverse_dct, down_sampling_variant, down_sampling_step)
    joined_channels_img = join_channels(y_inverse_dct, cb_up_sampled, cr_up_sampled)
    rgb_image = y_cb_cr_to_rgb(joined_channels_img, Y_CB_CR_MATRIX_INVERSE)
    unpadded_image = reverse_padding(rgb_image, original_rows, original_cols)
    show_images(unpadded_image, encoded_image_name + " - Decompressed", None, None)

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

    down_sampling_variant = eval(input("Down sampling variant: "))
    down_sampling_step = eval(input("Down sampling step: "))
    block_size = eval(input("Block size: "))
    quality_factor = eval(input("Quality factor: "))
    show_plots = False

    original_images = read_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION)
    if show_plots:
        show_images(original_images, None, None, None)
        jpeg_compress_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION, comp_img_dir, JPEG_QUALITY_RATES)

    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder((image_name, original_images[image_name]),
                         down_sampling_variant, down_sampling_step, block_size, quality_factor, show_plots=False)
        encoded_images[image_name] = (result[0], result[1], result[2], result[3], result[4], result[5], result[6])

    decoded_images = dict()
    for encoded_image_name in encoded_images.keys():
        data = encoded_images[encoded_image_name]
        result = decoder((encoded_image_name, data[0], data[1], data[2], data[3], data[4], data[5], data[6]))
        decoded_images[encoded_image_name] = result

        if image_equals(original_images[encoded_image_name], result):
            print("No diff")
        else:
            print("Diff")


if __name__ == '__main__':
    main()
