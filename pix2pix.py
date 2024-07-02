! pip install rasterio
import tensorflow as tf

import os
import pathlib
import time
import datetime
import glob
import rasterio
import numpy as np
import cv2

from matplotlib import pyplot as plt
%matplotlib inline
from IPython import display
def sar_preprop1(img):
    kernel = np.array([[-1, 1, -1],
                       [-1, 1,-1],
                       [-1, 1, -1]])
    processed_img= cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return processed_img


def sar_preprop2(img):
    kernel = np.ones((5,5),np.float32)/25
    processed_img= cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return processed_img

def sar_image(file_path):  #.tif file
    ds = rasterio.open(file_path)
    img = ds.read()
    img = cv2.medianBlur(img,5)
    img = img.reshape(img.shape[-1],img.shape[-2],img.shape[0])

    img1 = sar_preprop1(img)
    img2 = sar_preprop2(img)
    img = np.dstack((img,img1,img2))
    img = tf.cast(img, tf.float32)
    return img

def eo_image(file_path):
    img = None
    with rasterio.open(file_path) as ds:
        b1 = ds.read(1)[:-1,:-1]
        b2 = ds.read(2)[:-1,:-1]
        b3 = ds.read(3)[:-1,:-1]
        img = np.dstack((b1, b2, b3))
    img = tf.cast(img, tf.float32)
    return img

BUFFER_SIZE = 10000
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image
