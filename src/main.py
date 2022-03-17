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
from modules.metrics import *
from modules.jpeg_partial_codec import *


def main():
    """
    Main function
    """
    cwd = os.getcwd()
    orig_img_dir = cwd + ORIGINAL_IMAGE_DIRECTORY
    comp_img_dir = cwd + COMPRESSED_IMAGE_DIRECTORY

    down_sample_variant = input("Down sampling variant: ")
    if not down_sample_variant:
        down_sample_variant = "4:2:2"

    quality_factor = input("Quality factor: ")
    if not quality_factor:
        quality_factor = 50
    else:
        quality_factor = eval(quality_factor)

    show_plots = False

    original_images = read_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION)
    if show_plots:
        show_images(original_images, None, None, None)
        jpeg_compress_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION, comp_img_dir, JPEG_QUALITY_RATES)

    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder((image_name, original_images[image_name]),
                         down_sample_variant, IMAGE_SIZE_DIVISOR, quality_factor, show_plots=True)
        encoded_images[image_name] = (result[0], result[1], result[2], result[3], result[4], result[5])

    for encoded_image_name in encoded_images.keys():
        data = encoded_images[encoded_image_name]
        print("Decompressed image %s" % encoded_image_name)
        result, y_new = decoder((encoded_image_name, data[0], data[1], data[2], data[3], data[4]))
        y_old = encoded_images[encoded_image_name][5]
        print("Distortion metrics - Y channel")
        show_images(calc_error_image(y_old, y_new), encoded_image_name + " - Error - Y channel", GREY_CMAP, None)
        show_jpeg_metrics(y_old, y_new)
        print("--------------------")


if __name__ == '__main__':
    main()
