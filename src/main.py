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
from modules.jpeg_partial_codec import *
from modules.metrics import calc_error_image, show_jpeg_metrics


def read_config():
    config = dict()

    quality_factor = input("Quality factor: ")
    if not quality_factor:
        quality_factor = 75
    else:
        if ',' in quality_factor:
            quality_factor = quality_factor.replace(" ", "").split(",")
        else:
            quality_factor = eval(quality_factor)
    print(quality_factor)

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

    show_plots = input("Show plots?")
    if not show_plots:
        show_plots = False
    else:
        show_plots = True
    print(show_plots)

    verbose = input("Verbose?")
    if not verbose:
        verbose = True
    else:
        verbose = False
    print(verbose)

    config["quality_factor"] = quality_factor
    config["down_sample_variant"] = down_sample_variant
    config["interpolation_type"] = interpolation_type
    config["show_plots"] = show_plots
    config["verbose"] = verbose

    return config


def codec_run(original_images, config, output_file):
    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder(output_file, (image_name, original_images[image_name]),
                         config["down_sample_variant"], IMAGE_SIZE_DIVISOR, config["quality_factor"], config["interpolation_type"],
                         show_plots=config["show_plots"], verbose=config["verbose"])
        encoded_images[image_name] = (result[0], result[1], result[2], result[3], result[4], result[5])

    for encoded_image_name in encoded_images.keys():
        data = encoded_images[encoded_image_name]
        print("Decompressed image %s" % encoded_image_name)
        result, y_new = decoder((encoded_image_name, data[0], data[1], data[2], data[3], data[4], data[5]))
        image_old = original_images[encoded_image_name]
        print("Distortion metrics")
        if config["show_plots"]:
            show_images((calc_error_image(data[5], y_new)), encoded_image_name + " - Error - Y channel", GREY_CMAP, None)
        show_jpeg_metrics(image_old, result)
        print("--------------------")


def main():
    """
    Main function
    """
    cwd = os.getcwd()
    orig_img_dir = cwd + ORIGINAL_IMAGE_DIRECTORY
    comp_img_dir = cwd + COMPRESSED_IMAGE_DIRECTORY

    config = read_config()

    original_images = read_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION)
    if config["show_plots"]:
        show_images(original_images, None, None, None)
        jpeg_compress_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION, comp_img_dir, JPEG_QUALITY_RATES)

    output_file_name = os.getcwd() + OUTPUT_TXT_PATH + "-" + \
                                   str(config["quality_factor"]) + "-" + \
                                   config["down_sample_variant"].replace(":", "-") + ".txt"

    with open(output_file_name, "w") as output_file:
        if type(config["quality_factor"]) == list:
            for qual_factor in config["quality_factor"]:
                inner_config = config.copy()
                inner_config["quality_factor"] = int(qual_factor)
                codec_run(original_images, inner_config, output_file)
        else:
            codec_run(original_images, config, output_file)


if __name__ == '__main__':
    main()
