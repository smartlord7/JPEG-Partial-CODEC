import os
import matplotlib.colors as clr
from PIL import Image
from matplotlib import pyplot as plt


def _show_image(image, name, cmap, func):
    """
      Given one image,the title and a color map this function will show the image with the title and the applied colormap
      :param image: the image to show.
      :param name: the name of the image to show.
      :param cmap: the color map to be applied.
    """
    plt.figure()
    if name:
        plt.title(name)
    if func:
        plt.imshow(func(image), cmap)
    else:
        plt.imshow(image, cmap)
    plt.show()


def show_images(images, name, cmap, func):
    """
      Given one or more images,this function will show them in order
      :param images: the image(s) to show.
      :param name: the name of the image to show
      :return:
    """
    t = type(images)

    if t == dict:
        for image_name in images:
            _show_image(images[image_name], image_name, cmap, func)
    elif t == list:
        for image in images:
            _show_image(image, name, cmap, func)
    else:
        _show_image(images, name, cmap, func)


def read_images(directory, ext):
    """
          Given one directory and a file extension,this function will create a dictionary of the images in the directory.
          :param directory: the image(s) directory.
          :param ext: the image(s) extension.
          :return: dictionary with the images.
    """
    images = dict()

    for image_name in os.listdir(directory):
        if image_name.endswith(ext):
            image = plt.imread(directory + image_name)
            print("Read %s - shape: %s, type: %s" % (image_name, image.shape, image.dtype))
            images[image_name] = image

    return images


def read_images2(directory, ext):
    """
             Given one directory and a file extension,this function will create a dictionary of the images in the directory.
             :param directory: the image(s) directory.
             :param ext: the image(s) extension.
             :return: dictionary with the images.
    """
    images = dict()

    for image_name in os.listdir(directory):
        if image_name.endswith(ext):
            image = Image.open(directory + image_name)
            images[image_name] = image

    return images


def generate_linear_colormap(color_list):
    """
                            Generates the colormap.
                           :param color_list: list of colors.
                           :return: the colormap.
    """
    colormap = clr.LinearSegmentedColormap.from_list('cmap', color_list, N=256)

    return colormap
