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

from modules.jpeg import *
from modules.metrics import *
from modules.jpeg_partial_codec import *


def main():
    """
    Main function
    """
    cwd = os.getcwd()
    orig_img_dir = cwd + ORIGINAL_IMAGE_DIRECTORY
    comp_img_dir = cwd + COMPRESSED_IMAGE_DIRECTORY

    quality_factor = input("Quality factor: ")
    if not quality_factor:
        quality_factor = 75
    else:
        quality_factor = eval(quality_factor)
    print(quality_factor)

    down_sample_variant = input("Down sampling variant: ")
    if not down_sample_variant:
        down_sample_variant = "4:2:2"
    print(down_sample_variant)

    interpolation_type = input("Downsampling interpolation type: ")
    if "LINEAR".lower() in interpolation_type.lower():
        interpolation_type = cv2.INTER_LINEAR
    elif "CUBIC".lower() in interpolation_type.lower():
        interpolation_type = cv2.INTER_CUBIC
    elif "AREA".lower() in interpolation_type.lower():
        interpolation_type = cv2.INTER_AREA
    else:
        interpolation_type = None
    print(interpolation_type)

    show_plots = input("Show plots?")
    if not show_plots:
        show_plots = False
    else:
        show_plots = True
    print(show_plots)

    verbose = input("Verbose?")
    if not verbose:
        verbose = False
    else:
        verbose = True
    print(verbose)

    original_images = read_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION)
    if show_plots:
        show_images(original_images, None, None, None)
        jpeg_compress_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION, comp_img_dir, JPEG_QUALITY_RATES)

    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder((image_name, original_images[image_name]),
                         down_sample_variant, IMAGE_SIZE_DIVISOR, quality_factor, interpolation_type, show_plots=show_plots, verbose=verbose)
        encoded_images[image_name] = (result[0], result[1], result[2], result[3], result[4], result[5])

    for encoded_image_name in encoded_images.keys():
        data = encoded_images[encoded_image_name]
        print("Decompressed image %s" % encoded_image_name)
        result, y_new = decoder((encoded_image_name, data[0], data[1], data[2], data[3], data[4], data[5]))
        image_old = original_images[encoded_image_name]
        print("Distortion metrics")
        show_images((calc_error_image(data[5], y_new)), encoded_image_name + " - Error - Y channel", GREY_CMAP, None)
        show_jpeg_metrics(image_old, result)
        print("--------------------")


if __name__ == '__main__':
    main()
