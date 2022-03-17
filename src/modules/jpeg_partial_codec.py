from modules.util import *
from modules.const import *
from modules.image import *
from modules.jpeg_pipeline.dct import *
from modules.jpeg_pipeline.dpcm import *
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

    y, cb, cr = separate_channels(y_cb_cr_image)

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

    y_dct_blocks_8 = apply_dct_blocks_optimized(y, 8)
    cb_dct_blocks_8 = apply_dct_blocks_optimized(cb, 8)
    cr_dct_blocks_8 = apply_dct_blocks_optimized(cr, 8)

    joined_y_dct_blocks_8 = join_matrix_blockwise(y_dct_blocks_8)
    joined_cb_dct_blocks_8 = join_matrix_blockwise(cb_dct_blocks_8)
    joined_cr_dct_blocks_8 = join_matrix_blockwise(cr_dct_blocks_8)

    title_blocks_dct = image_name + " - DCT by blocks 8x8"
    if show_plots:
        show_images(joined_y_dct_blocks_8, title_blocks_dct + " - Y", GREY_CMAP, plot_f)
        show_images(joined_cb_dct_blocks_8, title_blocks_dct + " - Cb", GREY_CMAP, plot_f)
        show_images(joined_cr_dct_blocks_8, title_blocks_dct + " - Cr", GREY_CMAP, plot_f)

    #y_dct_blocks_64 = apply_dct_blocks_optimized(y, 64)
    #cb_dct_blocks_64 = apply_dct_blocks_optimized(cb, 64)
    #cr_dct_blocks_64 = apply_dct_blocks_optimized(cr, 64)

    #joined_y_dct_blocks_64 = join_matrix_blockwise(y_dct_blocks_64)
    #joined_cb_dct_blocks_64 = join_matrix_blockwise(cb_dct_blocks_64)
    #joined_cr_dct_blocks_64 = join_matrix_blockwise(cr_dct_blocks_64)

    #title_blocks_dct = image_name + " - DCT by blocks 64x64"
    #if show_plots:
        #show_images(joined_y_dct_blocks_64, title_blocks_dct + " - Y", GREY_CMAP, plot_f)
        #show_images(joined_cb_dct_blocks_64, title_blocks_dct + " - Cb", GREY_CMAP, plot_f)
        #show_images(joined_cr_dct_blocks_64, title_blocks_dct + " - Cr", GREY_CMAP, plot_f)

    y_blocks_quantized = apply_quantization(y_dct_blocks_8, quality_factor, JPEG_QUANTIZATION_Y)
    cb_blocks_quantized = apply_quantization(cb_dct_blocks_8, quality_factor, JPEG_QUANTIZATION_CB_CR)
    cr_blocks_quantized = apply_quantization(cr_dct_blocks_8, quality_factor, JPEG_QUANTIZATION_CB_CR)

    title_blocks_quantized = image_name + " - DCT by blocks 8x8 " +\
                             " w/quantization qual. " + str(quality_factor)
    if show_plots:
        show_images(join_matrix_blockwise(y_blocks_quantized), title_blocks_quantized + " - Y", GREY_CMAP, plot_f)
        show_images(join_matrix_blockwise(cb_blocks_quantized), title_blocks_quantized + " - Cb", GREY_CMAP, plot_f)
        show_images(join_matrix_blockwise(cr_blocks_quantized), title_blocks_quantized + " - Cr", GREY_CMAP, plot_f)

    y_blocks_dpcm = apply_dpcm_encoding(y_blocks_quantized)
    cb_blocks_dpcm = apply_dpcm_encoding(cb_blocks_quantized)
    cr_blocks_dpcm = apply_dpcm_encoding(cr_blocks_quantized)

    return (y_blocks_dpcm, cb_blocks_dpcm, cr_blocks_dpcm), n_rows, \
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

    y_idpcm = apply_dpcm_decoding(y)
    cb_idpcm = apply_dpcm_decoding(cb)
    cr_idpcm = apply_dpcm_decoding(cr)
    y_dequantized = apply_inverse_quantization(y_idpcm, quality_factor, JPEG_QUANTIZATION_Y)
    cb_dequantized = apply_inverse_quantization(cb_idpcm, quality_factor, JPEG_QUANTIZATION_CB_CR)
    cr_dequantized = apply_inverse_quantization(cr_idpcm, quality_factor, JPEG_QUANTIZATION_CB_CR)
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
