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
        quality_factor = {75}
    else:
        if ',' in quality_factor:
            quality_factor = set(quality_factor.replace(" ", "").split(","))
        else:
            quality_factor = {eval(quality_factor)}
    print(quality_factor)

    block_size = input("Block size(s): ")
    if not block_size:
        block_size = {8}
    else:
        if ',' in block_size:
            block_size = set(block_size.replace(" ", "").split(","))
        else:
            block_size = {eval(block_size)}
    print(block_size)

    down_sample_variant = input("Down sampling variant(s): ")
    if not down_sample_variant:
        down_sample_variant = {"4:2:0"}
    else:
        if ',' in down_sample_variant:
            down_sample_variant = set(down_sample_variant.replace(" ", "").split(","))
        else:
            down_sample_variant = {down_sample_variant}
    print(down_sample_variant)

    interpolation_type = input("Down sample interpolation type(s): ")
    if not interpolation_type:
        interpolation_type = {cv2.INTER_AREA}
    else:
        if ',' in interpolation_type:
            interpolation_type = set(interpolation_type.replace(" ", "").split(","))
        else:
            interpolation_type = {eval(interpolation_type)}
    print(interpolation_type)

    save_plots = input("Save plots?")
    if not save_plots:
        save_plots = False
    else:
        save_plots = True
    print(save_plots)

    verbose = input("Verbose?")
    if not verbose:
        verbose = False
    else:
        verbose = True
    print(verbose)

    config["quality_factor"] = quality_factor
    config["down_sample_variant"] = down_sample_variant
    config["block_size"] = block_size
    config["interpolation_type"] = interpolation_type
    config["save_plots"] = save_plots
    config["verbose"] = verbose

    return config


def codec_run(original_images, config):
    encoded_images = dict()

    for image_name in original_images.keys():
        result = encoder((image_name, original_images[image_name]),
                         config["down_sample_variant"], int(config["block_size"]), config["quality_factor"], interpolation_type=config["interpolation_type"],
                         save_plots=config["save_plots"], verbose=config["verbose"])
        encoded_images[image_name] = result

    for encoded_image_name in encoded_images.keys():
        data = encoded_images[encoded_image_name]
        result, output_file = decoder(encoded_image_name, data, save_plots=config["save_plots"], verbose=config["verbose"])
        image_old = original_images[encoded_image_name]
        show_jpeg_metrics(image_old, result, output_file)


def main():
    """
    Main function
    """

    config = read_config()
    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    #jpeg_compress_images(orig_img_dir, ORIGINAL_IMAGE_EXTENSION, comp_img_dir, JPEG_QUALITY_RATES)

    for qual_fact in config["quality_factor"]:
        for ds_var in config["down_sample_variant"]:
            for b_size in config["block_size"]:
                for interp_type in config["interpolation_type"]:
                    inner_config = config.copy()
                    inner_config["quality_factor"] = float(qual_fact)
                    inner_config["down_sample_variant"] = ds_var
                    inner_config["block_size"] = int(b_size)
                    inner_config["interpolation_type"] = int(interp_type)
                    codec_run(original_images, inner_config)


if __name__ == '__main__':
    main()
