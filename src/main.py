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

from modules.jpeg_partial_codec import *


def read_config():
    config = dict()

    quality_factor = input("Quality factor(s): ")
    if not quality_factor:
        quality_factor = 75
    else:
        if ',' in quality_factor:
            quality_factor = quality_factor.replace(" ", "").split(",")
        else:
            quality_factor = eval(quality_factor)
    print(quality_factor)

    block_size = input("Block size: ")
    if not block_size:
        block_size = 8
    print(block_size)

    down_sample_variant = input("Down sampling variant: ")
    if not down_sample_variant:
        down_sample_variant = "4:2:2"
    print(down_sample_variant)

    interpolation_type = input("Down sampling interpolation type: ")
    if "LINEAR".lower() in interpolation_type.lower():
        interpolation_type = cv2.INTER_LINEAR
    elif "CUBIC".lower() in interpolation_type.lower():
        interpolation_type = cv2.INTER_CUBIC
    elif "AREA".lower() in interpolation_type.lower():
        interpolation_type = cv2.INTER_AREA
    else:
        interpolation_type = None
    print(interpolation_type)

    show_plots = input("Save plots?")
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

    config["quality_factor"] = quality_factor
    config["block_size"] = block_size
    config["down_sample_variant"] = down_sample_variant
    config["interpolation_type"] = interpolation_type
    config["show_plots"] = show_plots
    config["verbose"] = verbose

    return config


def codec_run(original_images, config):
    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder((image_name, original_images[image_name]),
                         config["down_sample_variant"], config["block_size"], config["quality_factor"], interpolation_type=config["interpolation_type"],
                         show_plots=config["show_plots"], verbose=config["verbose"])
        encoded_images[image_name] = result

    for encoded_image_name in encoded_images.keys():
        data = encoded_images[encoded_image_name]
        result, output_file = decoder(encoded_image_name, data, show_plots=config["show_plots"], verbose=config["verbose"])
        image_old = original_images[encoded_image_name]
        show_jpeg_metrics(image_old, result, output_file)


def main():
    """
    Main function
    """

    config = read_config()
    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    #jpeg_compress_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION, comp_img_dir, JPEG_QUALITY_RATES)

    if type(config["quality_factor"]) == list:
        for qual_factor in config["quality_factor"]:
            inner_config = config.copy()
            inner_config["quality_factor"] = int(qual_factor)
            codec_run(original_images, inner_config)
    else:
        codec_run(original_images, config)


if __name__ == '__main__':
    main()
