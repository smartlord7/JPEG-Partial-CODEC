import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
import os


def show_images(images):
    for image_name in images.keys():
        plt.figure()
        plt.imshow(images[image_name])
        plt.show()


def read_images(directory, ext):
    images = dict()

    for image_name in os.listdir(directory):
        if image_name.endswith(ext):
            image = plt.imread(directory + image_name)
            print("Read %s - shape: %s, type: %s" % (image_name, image.shape, image.dtype))
            images[image_name] = image

    return images


def read_images2(directory, ext):
    images = dict()

    for image_name in os.listdir(directory):
        if image_name.endswith(ext):
            image = Image.open(directory + image_name)
            images[image_name] = image

    return images


def separate_rgb():
    img = Image.open('C:/Users/Ventura/PycharmProjects/pythonProject/original_img/barn_mountains.bmp')
    array = np.array(img)
    r, g, b = array.copy(), array.copy(), array.copy()
    r[:, :, (1, 2)] = 0
    g[:, :, (0, 2)] = 0
    b[:, :, (0, 1)] = 0
    img_rgb = np.concatenate((r, g, b))
    plt.figure(figsize=(30, 30))
    plt.imshow(img_rgb)
    plt.show()
    transformer(img_rgb)

def transformer(rgb):
    tform= np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    ycbcr = rgb.dot(tform.T)
    ycbcr[:, :, [1, 2]] += 128
    #np.uint8(ycbcr)
    plt.figure(figsize=(30, 30))
    plt.imshow(ycbcr.astype(np.uint8))
    plt.show()


def jpeg_compress_images(directory, ext, out_dir, quality_rates):
    images = read_images2(directory, ext)
    fig, axis = plt.subplots(len(images), len(quality_rates))
    i = 0

    for image_name in images.keys():
        j = 0
        for quality_rate in quality_rates:
            compressed_image_name = image_name.replace(ext, "") + str(quality_rate) + ".jpg"
            compress_image_path = out_dir + "/" + compressed_image_name
            images[image_name].save(compress_image_path, quality=quality_rate)
            image = plt.imread(compress_image_path)
            axis[i, j].imshow(image)
            axis[i, j].set_title(compressed_image_name, fontsize=10)

            j += 1

        i += 1

    plt.show()


def encoder(image, params):
    pass


def decoder(encoded_image):
    pass


def main():
    CWD = os.getcwd()
    ORIGINAL_IMAGE_DIRECTORY = CWD + '/original_img/'
    ORIGINAL_IMAGE_EXTENSION = '.bmp'
    COMPRESSED_IMAGE_DIRECTORY = CWD + "\\jpeg_compressed_img"
    JPEG_QUALITY_RATES = [25, 50, 75]

    original_images = dict()

    original_images = read_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION)
    # show_images(original_images)
    # jpeg_compress_images(ORIGINAL_IMAGE_DIRECTORY, ORIGINAL_IMAGE_EXTENSION, COMPRESSED_IMAGE_DIRECTORY, JPEG_QUALITY_RATES)
    separate_rgb()


if __name__ == '__main__':
    main()
