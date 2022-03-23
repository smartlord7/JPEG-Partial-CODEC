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

import os
import numpy as np
from PIL import Image
import matplotlib.colors as clr
from matplotlib import pyplot as plt

OUTPUT_IMG_PATH = os.getcwd() + "\\resources\\img\\plots\\"


def _show_image(image, name, cmap, func):
    """
      Given one image,the title and a color map this function will show the image with the title and the applied colormap
      :param image: the image to show.
      :param name: the name of the image to show.
      :param cmap: the color map to be applied.
      :param func: the function of the image.
    """
    plt.figure()
    if name:
        plt.title(name)
    if func:
        plt.imshow(func(image), cmap)
    else:
        plt.imshow(image, cmap)

    if name:
        plt.savefig(OUTPUT_IMG_PATH + name + ".png")
    else:
        plt.savefig(OUTPUT_IMG_PATH + "-.png")
    plt.show()


def show_images(images, name, cmap, func):
    """
      Given one or more images,this function will show them in order
      :param images: the image(s) to show.
      :param name: the name of the image to show
      :param cmap: the color map to be applied.
      :param func: the function of the image.
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


def image_equals(original_image, decoded_image):
    """
    Verifies if the images are equal.
    :param original_image: original image.
    :param decoded_image: decoded image.
    :return: if the image is equal or no.
    """
    return np.allclose(original_image, decoded_image)