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
