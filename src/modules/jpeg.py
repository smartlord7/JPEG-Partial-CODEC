from matplotlib import pyplot as plt
from modules.image import read_images2


def jpeg_compress_images(directory, ext, out_dir, quality_rates):
    """
                        Compresses images.
                        :param directory: images directory.
                        :param ext: images extension.
                        :param out_dir: output directory.
                        :param quality_rates: the quality rates of the compression.
                        :return:
    """
    images = read_images2(directory, ext)
    fig, axis = plt.subplots(len(images), len(quality_rates))
    i = 0

    for image_name in images.keys():
        j = 0
        for quality_rate in quality_rates:
            compressed_image_name = image_name.replace(ext, "") + str(quality_rate) + ".jpg"
            compress_image_path = out_dir + "\\" + compressed_image_name
            images[image_name].save(compress_image_path, quality=quality_rate)
            image = plt.imread(compress_image_path)
            axis[i, j].imshow(image)
            axis[i, j].set_title(compressed_image_name, fontsize=10)

            j += 1

        i += 1

    plt.show()
