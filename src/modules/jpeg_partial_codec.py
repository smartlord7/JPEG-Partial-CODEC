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


from time import perf_counter
from modules.util import *
from modules.metrics import *
from modules.jpeg_pipeline.dct import *
from modules.jpeg_pipeline.dpcm import *
from modules.jpeg_pipeline.padding import *
from modules.jpeg_pipeline.y_cb_cr import *
from modules.jpeg_pipeline.sampling import *
from modules.jpeg_pipeline.quantization import *


def encoder(image_data: tuple, down_sample_variant: str, block_size: int, quality_factor: float,
            interpolation_type: int = cv2.INTER_CUBIC, save_plots: bool = False, verbose: bool = False):
    """
    Enconder function.
    :param image_data: a tuple containing the target image data.
    :param quality_factor: the quality factor to use (from 0 to 100).
    :param down_sample_variant: The variant of the down sample (4:2:2, 4:2:0, 4:4:4, ...).
    :param block_size: the size of the block.
    :param interpolation_type: the type of interpolation to use (e.g. LINEAR; CUBIC; AREA).
    :param save_plots: flag that enables plotting.
    :param verbose: flag that enables the log output.
    :return: the encoded image and the y copy.
    """
    image_name = image_data[0]
    image_matrix = image_data[1]
    n_rows = image_matrix.shape[0]
    n_cols = image_matrix.shape[1]
    n = n_rows * n_cols
    total_time = int()
    img_id = image_name + "-Q" + str(quality_factor) + "D" + down_sample_variant.replace(":", "") + "B" + str(
        block_size) + "I" + str(interpolation_type)

    mkdir_if_not_exists(OUTPUT_TXT_ENCODER_PATH)
    output_file_path = OUTPUT_TXT_ENCODER_PATH + img_id + OUTPUT_LOG_EXTENSION
    output_file = open(output_file_path, "w")
    img_dir_path = OUTPUT_IMG_ENCODER_PATH + img_id + OUTPUT_LOG_COMPRESSION_SUFFIX
    mkdir_if_not_exists(img_dir_path)

    out(output_file, "------------------------------------------------")
    out(output_file, "Compressing %s (shape: %s) with quality factor of %.2f%% and %s down sampling..." % (
        image_name, image_matrix.shape, quality_factor, down_sample_variant))

    cb_fac, cr_fac, s = parse_down_sample_variant(down_sample_variant)
    s_cols = int()
    s_rows = int()

    if cb_fac == cr_fac:
        s_cols = s
        s_rows = 1
    elif cr_fac == 0:
        s_cols = s
        s_rows = s

    timer = perf_counter()
    padded_image = apply_padding(image_matrix, s_rows * block_size, s_cols * block_size)
    total_time += perf_counter() - timer

    new_shape = padded_image.shape
    added_rows = str(new_shape[0] - n_rows)
    added_cols = str(new_shape[1] - n_cols)

    if verbose:
        out(output_file, "Applied padding of %s rows and %s columns" % (added_rows, added_cols))

    if save_plots:
        save_images(padded_image, img_id + "-Padded+" + added_rows + "+" + added_cols, None, None, img_dir_path)

    r, g, b = separate_channels(padded_image)
    if save_plots:
        save_images(r, img_id + "-RRed", RED_CMAP, None, img_dir_path)
        save_images(g, img_id + "-GGreen", GREEN_CMAP, None, img_dir_path)
        save_images(b, img_id + "-BBlue", BLUE_CMAP, None, img_dir_path)

    calc_entropic_stats(image_name, [r, g, b], ["R", "G", "B"], "Original", output_file, img_dir_path)

    timer = perf_counter()
    y_cb_cr_image = rgb_to_y_cb_cr(padded_image, Y_CB_CR_MATRIX)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Converted RGB to YCbCr")

    timer = perf_counter()
    y, cb, cr = separate_channels(y_cb_cr_image)
    total_time += perf_counter() - timer

    calc_entropic_stats(image_name, [float_to_uint8(y), float_to_uint8(cb), float_to_uint8(cr)], ["Y", "Cb", "Cr"],
                        "YCbCr", output_file, img_dir_path)

    if verbose:
        out(output_file, "Separated Y, Cb and Cr channels")

    y_copy = y

    if save_plots:
        save_images(y, img_id + "-YGrey", GREY_CMAP, None, img_dir_path)
        save_images(cb, img_id + "-CbGrey", GREY_CMAP, None, img_dir_path)
        save_images(cr, img_id + "-CrGrey", GREY_CMAP, None, img_dir_path)

    timer = perf_counter()
    cb, cr = down_sample(cb, cr, down_sample_variant, interpolation_type=interpolation_type)
    total_time += perf_counter() - timer

    n_new = cb.shape[0] * cb.shape[1]

    if verbose:
        out(output_file, "Down sampled Cb and Cr channels using %s - shape: %s - compression ratio: %.2f%%" %
            (down_sample_variant, cb.shape, ((n - n_new) / n * 100)))

    if save_plots:
        save_images(cb, img_id + "-CbDownSampled", GREY_CMAP, plot_f, img_dir_path)
        save_images(cr, img_id + "-CrDownSampled", GREY_CMAP, plot_f, img_dir_path)

    y_dct_total = apply_dct(y)
    cb_dct_total = apply_dct(cb)
    cr_dct_total = apply_dct(cr)

    if save_plots:
        save_images(y_dct_total, img_id + "-YTotalDCT", GREY_CMAP, plot_f, img_dir_path)
        save_images(cb_dct_total, img_id + "-CbTotalDCT", GREY_CMAP, plot_f, img_dir_path)
        save_images(cr_dct_total, img_id + "-CrTotalDCT", GREY_CMAP, plot_f, img_dir_path)

    y_idct_total = apply_inverse_dct(y_dct_total)
    cb_idct_total = apply_inverse_dct(cb_dct_total)
    cr_idct_total = apply_inverse_dct(cr_dct_total)

    if save_plots:
        save_images(y_idct_total, img_id + "-YTotalIDCT", GREY_CMAP, None, img_dir_path)
        save_images(cb_idct_total, img_id + "-CbTotalIDCT", GREY_CMAP, None, img_dir_path)
        save_images(cr_idct_total, img_id + "-CrTotalIDCT", GREY_CMAP, None, img_dir_path)

    timer = perf_counter()
    y_dct_blocks = apply_dct_blocks_optimized(y, block_size)
    cb_dct_blocks = apply_dct_blocks_optimized(cb, block_size)
    cr_dct_blocks = apply_dct_blocks_optimized(cr, block_size)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied DCT in blocks of %d" % block_size)

    joined_y_dct_blocks = join_matrix_blockwise(y_dct_blocks)
    joined_cb_dct_blocks = join_matrix_blockwise(cb_dct_blocks)
    joined_cr_dct_blocks = join_matrix_blockwise(cr_dct_blocks)

    if save_plots:
        save_images(joined_y_dct_blocks, img_id + "-YDCT", GREY_CMAP, plot_f, img_dir_path)
        save_images(joined_cb_dct_blocks, img_id + "-CbDCT", GREY_CMAP, plot_f, img_dir_path)
        save_images(joined_cr_dct_blocks, img_id + "-CrDCT", GREY_CMAP, plot_f, img_dir_path)

    calc_entropic_stats(image_name, [float_to_uint8(joined_y_dct_blocks), float_to_uint8(joined_cb_dct_blocks),
                                     float_to_uint8(joined_cr_dct_blocks)],
                        ["Y", "Cb", "Cr"], "DCT %dx%d" % (block_size, block_size), output_file, img_dir_path)

    timer = perf_counter()
    y_blocks_quantized = apply_quantization(y_dct_blocks, quality_factor, JPEG_QUANTIZATION_Y)
    cb_blocks_quantized = apply_quantization(cb_dct_blocks, quality_factor, JPEG_QUANTIZATION_CB_CR)
    cr_blocks_quantized = apply_quantization(cr_dct_blocks, quality_factor, JPEG_QUANTIZATION_CB_CR)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied quantization using quality factor %.2f%%" % quality_factor)

    joined_y_blocks_quantized = join_matrix_blockwise(y_blocks_quantized)
    joined_cb_blocks_quantized = join_matrix_blockwise(cb_blocks_quantized)
    joined_cr_blocks_quantized = join_matrix_blockwise(cr_blocks_quantized)

    if save_plots:
        save_images(joined_y_blocks_quantized, img_id + "-YQuantized", GREY_CMAP, plot_f, img_dir_path)
        save_images(joined_cb_blocks_quantized, img_id + "-CbQuantized", GREY_CMAP, plot_f, img_dir_path)
        save_images(joined_cr_blocks_quantized, img_id + "-CrQuantized", GREY_CMAP, plot_f, img_dir_path)

    calc_entropic_stats(image_name,
                        [float_to_uint8(joined_y_blocks_quantized), float_to_uint8(joined_cb_blocks_quantized),
                         float_to_uint8(joined_cr_blocks_quantized)],
                        ["Y", "Cb", "Cr"], "DCT %dx%d Quantized" % (block_size, block_size), output_file, img_dir_path)

    timer = perf_counter()
    y_blocks_dpcm = apply_dpcm_encoding(y_blocks_quantized)
    cb_blocks_dpcm = apply_dpcm_encoding(cb_blocks_quantized)
    cr_blocks_dpcm = apply_dpcm_encoding(cr_blocks_quantized)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied DPCM")

    joined_y_blocks_dpcm = join_matrix_blockwise(y_blocks_dpcm)
    joined_cb_blocks_dpcm = join_matrix_blockwise(cb_blocks_dpcm)
    joined_cr_blocks_dpcm = join_matrix_blockwise(cr_blocks_dpcm)

    if save_plots:
        save_images(joined_y_blocks_dpcm, img_id + "-YDPCM", GREY_CMAP, plot_f, img_dir_path)
        save_images(joined_cb_blocks_dpcm, img_id + "-CbDPCM", GREY_CMAP, plot_f, img_dir_path)
        save_images(joined_cr_blocks_dpcm, img_id + "-CrDPCM", GREY_CMAP, plot_f, img_dir_path)

    calc_entropic_stats(image_name,
                        [float_to_uint8(joined_y_blocks_dpcm), float_to_uint8(joined_cb_blocks_dpcm),
                         float_to_uint8(joined_cr_blocks_dpcm)],
                        ["Y", "Cb", "Cr"], "DCT DC DPCM", output_file, img_dir_path)

    if verbose:
        out(output_file, "Elapsed compression time: %.3fms" % total_time)
        out(output_file, "----------------------------------\n")

    output_file.close()

    return (y_blocks_dpcm, cb_blocks_dpcm, cr_blocks_dpcm), n_rows, \
           n_cols, down_sample_variant, quality_factor, block_size, interpolation_type, y_copy


def decoder(encoded_image_name: str, encoded_image_data, verbose: bool = False, save_plots: bool = False):
    """
                                           Decode function.
                                           :param encoded_image_name:
                                           :param verbose:
                                           :param save_plots:
                                           :param encoded_image_data: the image to decode.
                                           :return: the decoded image and the y copy error.
    """
    encoded_image = encoded_image_data[0]
    original_rows = encoded_image_data[1]
    original_cols = encoded_image_data[2]
    down_sample_variant = encoded_image_data[3]
    quality_factor = encoded_image_data[4]
    block_size = encoded_image_data[5]
    interpolation_type = encoded_image_data[6]
    y_original = encoded_image_data[7]
    total_time = int()
    img_id = encoded_image_name + "-Q" + str(quality_factor) + "D" + down_sample_variant.replace(":", "") + "B" + str(
        block_size) + "I" + str(interpolation_type)

    mkdir_if_not_exists(OUTPUT_TXT_DECODER_PATH)
    output_file_path = OUTPUT_TXT_DECODER_PATH + img_id + OUTPUT_LOG_EXTENSION
    output_file = open(output_file_path, "w")
    img_dir_path = OUTPUT_IMG_DECODER_PATH + img_id + OUTPUT_LOG_DECOMPRESSION_SUFFIX
    mkdir_if_not_exists(img_dir_path)

    out(output_file, "----------------------------------")
    out(output_file, "Decompressing %s with quality factor of %.2f%% and %s down sampling..." % (
        encoded_image_name, quality_factor, down_sample_variant))

    y = encoded_image[0]
    cb = encoded_image[1]
    cr = encoded_image[2]

    timer = perf_counter()
    y_idpcm = apply_dpcm_decoding(y)
    cb_idpcm = apply_dpcm_decoding(cb)
    cr_idpcm = apply_dpcm_decoding(cr)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied inverse DPCM encoding")

    if save_plots:
        save_images(join_matrix_blockwise(y_idpcm), img_id + "-YIDPCM", GREY_CMAP, plot_f, img_dir_path)
        save_images(join_matrix_blockwise(cb_idpcm), img_id + "-CbIDPCM", GREY_CMAP, plot_f, img_dir_path)
        save_images(join_matrix_blockwise(cr_idpcm), img_id + "-CrIDPCM", GREY_CMAP, plot_f, img_dir_path)

    timer = perf_counter()
    y_dequantized = apply_inverse_quantization(y_idpcm, quality_factor, JPEG_QUANTIZATION_Y)
    cb_dequantized = apply_inverse_quantization(cb_idpcm, quality_factor, JPEG_QUANTIZATION_CB_CR)
    cr_dequantized = apply_inverse_quantization(cr_idpcm, quality_factor, JPEG_QUANTIZATION_CB_CR)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied dequantization")

    if save_plots:
        save_images(join_matrix_blockwise(y_idpcm), img_id + "-YDequantized", GREY_CMAP, plot_f, img_dir_path)
        save_images(join_matrix_blockwise(cb_idpcm), img_id + "-CbDequantized", GREY_CMAP, plot_f, img_dir_path)
        save_images(join_matrix_blockwise(cr_idpcm), img_id + "-CrDequantized", GREY_CMAP, plot_f, img_dir_path)

    timer = perf_counter()
    y_inverse_dct = apply_inverse_dct_blocks_optimized(y_dequantized)
    cb_inverse_dct = apply_inverse_dct_blocks_optimized(cb_dequantized)
    cr_inverse_dct = apply_inverse_dct_blocks_optimized(cr_dequantized)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied inverse DCT")

    if save_plots:
        save_images(y_inverse_dct, img_id + "-YIDCT", GREY_CMAP, None, img_dir_path)
        save_images(cb_inverse_dct, img_id + "-CbIDCT", GREY_CMAP, None, img_dir_path)
        save_images(cr_inverse_dct, img_id + "-CrIDCT", GREY_CMAP, None, img_dir_path)
        save_images((calc_error_matrix(y_original, y_inverse_dct)), img_id + "YError", GREY_CMAP, None, img_dir_path)

    timer = perf_counter()
    cb_up_sampled, cr_up_sampled = up_sample(cb_inverse_dct, cr_inverse_dct, down_sample_variant,
                                             interpolation_type=cv2.INTER_AREA)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied up sampling using %s" % down_sample_variant)

    if save_plots:
        save_images(cb_up_sampled, img_id + "-CbUpSampled", GREY_CMAP, None, img_dir_path)
        save_images(cr_up_sampled, img_id + "-CrUpSampled", GREY_CMAP, None, img_dir_path)

    timer = perf_counter()
    joined_channels_img = join_channels(y_inverse_dct, cb_up_sampled, cr_up_sampled)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Joined Y, Cb and Cr channels")

    timer = perf_counter()
    rgb_image = y_cb_cr_to_rgb(joined_channels_img, Y_CB_CR_MATRIX_INVERSE)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Converted YCbCr to RGB")

    if save_plots:
        save_images(rgb_image, img_id + "-RGB", None, None, img_dir_path)

    timer = perf_counter()
    unpadded_image = inverse_padding(rgb_image, original_rows, original_cols)
    total_time += perf_counter() - timer

    if verbose:
        out(output_file, "Applied inverse padding")

    if save_plots:
        save_images(unpadded_image, img_id + "-Unpadded", None, None, img_dir_path)

    out(output_file, "Elapsed decompression time: %.3fms" % total_time)
    out(output_file, "----------------------------------\n")

    save_images(unpadded_image, img_id + "-Decompressed", None, None, img_dir_path)
    decoded_image = unpadded_image

    return decoded_image, output_file
