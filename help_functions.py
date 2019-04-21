import logging
import os
import random
import re

import cv2
import numpy as np

import config
import retinex
from config import body_image_resize, face_image_resize


def unzip(arr):
    r = [list(t) for t in zip(*arr)]
    if len(r) != 2:
        return [], []
    return r[0], r[1]


def prepare_face(image):
    image = cv2.resize(image, config.face_image_resize)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    return image


def prepare_body(image):
    image = cv2.resize(image, body_image_resize)
    return image


def prepare_body_retinex(image):
    image = prepare_body(image)
    image = retinex.automatedMSRCR(
        image,
        [15, 80, 250]
    )
    return image


def prepare_face_random_resize(image):
    r = random.randint(face_image_resize[0] / 2, face_image_resize[0])
    image = cv2.resize(image, (r, r))
    return prepare_face(image)


def identity(image):
    return image


# fetches image from location
def fetch_images(path, file_type='.jpg', preprocess=identity):
    result = []
    for file in os.listdir(path):
        if (file.endswith(file_type) or file_type == '') and os.path.isfile(os.path.join(path, file)):
            image, _ = load_image(path, file, preprocess)
            result.append((image, file),)
    return result


saved_images = {}


# returns list of tuples containing loaded images and their file name
def load_all_images(path, file_type='.jpg', preprocess=identity):
    key = path + '#' + file_type + '#' + str(id(preprocess))
    images = saved_images.get(key)
    if images is None:
        images = fetch_images(path, file_type, preprocess)
        saved_images[key] = images
        logging.debug('Images loaded from path')
    else:
        logging.debug('Images loaded from memory')
    return images


# returns tuple containing loaded image and its file name
def load_image(path, file, preprocess=identity):
    if file.lower().endswith('.pgm'):
        image = cv2.imread(os.path.join(path, file), -1)
    else:
        image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
    image = preprocess(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    return image, file


# files <- iterable, contains names of images to load
def load_images(path, files, preprocess=identity):
    result = []
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            image = load_image(path, file, preprocess)
            result.append(image)
    logging.debug('LOADING: Specified images loaded (' + str(len(result)) + ')')
    return result


def info_from_image_name(file):
    match = re.search('^([0-9]+)_c([0-9]+)_f([0-9]+)', file)
    person_id = int(match.group(1))
    camera = int(match.group(2))
    frame = int(match.group(3))
    return person_id, camera, frame


def fetch_data(path):
    return np.loadtxt(path, delimiter=' ')


saved_data = {}


def load_data(file):
    key = file
    data = saved_data.get(key)
    if data is None:
        data = fetch_data(file)
        saved_data[key] = data
        logging.debug('LOADING: Data loaded from path')
    else:
        logging.debug('LOADING: Data loaded from memory')
    return data


def _build_info(images, n_person_id):
    info = np.empty((len(images), 3), dtype=object)
    i = 0
    # info: person_id, camera_id, file_name
    for image, file in images:
        person_id, camera_id, _ = info_from_image_name(file)
        info[i] = np.array([person_id, camera_id, image], dtype=object)
        i += 1
    # info = info[:i, :]
    # get n_person_id unique ids and ignore all the other people
    unique_ids = np.unique(info[:, 0])
    np.random.shuffle(unique_ids)
    n_person_id = min(n_person_id, unique_ids.shape[0])
    unique_ids = unique_ids[:n_person_id]
    info = info[list(map(lambda el: el in unique_ids, info[:, 0])), :]

    return info, unique_ids
