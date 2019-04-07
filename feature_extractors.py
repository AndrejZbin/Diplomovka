import cv2
import math
import numpy as np

import help_functions
import retinex

import os

import logging

from config import *


# saves features of images to 'save_location' file
# format: features... personID cameraID
def save_features(images, save_location):
    f = open(save_location, 'w')
    for (image, file) in images:
            features = extract_divide(image)
            person_id, camera_id, _ = help_functions.info_from_image_name(file)
            f.write(' '.join(map(str, features)) + (' ' + str(person_id) + ' ' + str(camera_id) + '\n'))
    f.close()


# uses scale invariant local ternary patter and HSV 8x8x8 on 10x10 patches
def extract_divide(image, size=10):
    if size != 10:
        raise NotImplemented()
    width, height = 65, 155
    image = cv2.resize(image, (width, height))
    image = retinex.automatedMSRCR(
        image,
        [15, 80, 250]
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    result = []

    siltp1 = SILTP(gray, 0.3, 4, 3)
    siltp2 = SILTP(gray, 0.3, 4, 3)

    for y in range(0, height-size+1, 5):
        histograms = []
        for x in range(0, width-size+1, 5):
            hsv_ = hsv[y:y+size, x:x+size, :]
            histLTP1, _ = np.histogram(siltp1[y:y+size, x:x+size].flatten(), bins=81)
            histLTP2, _ = np.histogram(siltp2[y:y+size, x:x+size].flatten(), bins=81)
            histHSV, _ = np.histogramdd(hsv_.reshape(hsv_.shape[0] * hsv_.shape[1], hsv_.shape[2]), bins=8)

            histograms.append(np.concatenate([histLTP1, histLTP2, histHSV.flatten()]))

        histograms = np.array(histograms, int)
        result.append(np.amax(histograms, axis=0))
    return np.array(result, int).flatten()


# scale invariant local ternary patter
def SILTP(image, threshold=0.3, neighbors=8, radius=1):
    points = [(radius * math.sin(math.pi * 2 * i / neighbors),
               radius * math.cos(math.pi * 2 * i / neighbors)) for i in range(neighbors)]
    height, width = np.size(image, 0), np.size(image, 1)

    def scale(v, vn, s):
        if vn > (1+s) * v:
            return 2
        if vn < (1-s) * v:
            return 1
        return 0

    pattern = np.empty((height, width), int)
    for y in range(height):
        for x in range(width):
            sumpixel = 0
            for i, (py, px) in enumerate(points):
                x_ = math.floor(x + px + 0.5)
                y_ = math.floor(y + py + 0.5)
                if x_ < 0 or x_ >= width or y_ < 0 or y_ >= height:
                    continue
                sumpixel += scale(image[y][x], image[y_][x_], threshold) * math.pow(3, i)
            pattern[y][x] = sumpixel
    return pattern


if True:
    train_images = help_functions.load_all_images(train_folder, preprocess=help_functions.resize_and_retinex)
    save_features(train_images, train_output)

    test_images = help_functions.load_all_images(test_folder, preprocess=help_functions.resize_and_retinex)
    save_features(test_images, test_output)

    query_images = help_functions.load_all_images(query_folder, preprocess=help_functions.resize_and_retinex)
    save_features(query_images, query_output)
