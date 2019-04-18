import os
import cv2
import re
import numpy as np

import logging
import random

from config import test_folder, train_folder, body_image_resize, face_resize

import retinex


def unzip(arr):
    r = [list(t) for t in zip(*arr)]
    if len(r) != 2:
        return [], []
    return r[0], r[1]


def resize_and_retinex_body(image):
    image = cv2.resize(image, body_image_resize)
    image = retinex.automatedMSRCR(
        image,
        [15, 80, 250]
    )
    return image


def resize_body(image):
    image = cv2.resize(image, body_image_resize)
    return image


def resize_face(image):
    image = cv2.resize(image, face_resize)
    return image


def identity(image):
    return image


# fetches image from location
# returns tuple containing loaded image and its file name
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


def load_image(path, file, preprocess=identity):
    if file.lower().endswith('.pgm'):
        image = cv2.imread(os.path.join(path, file), -1)
        image = np.expand_dims(image, axis=2)
    else:
        image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
    image = preprocess(image)
    return image, file


# files <- iterable, contains names of images to load
def load_images(path, files, preprocess=identity):
    result = []
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            image = load_image(path, file, preprocess)
            result.append(image)
    logging.debug('Specified images loaded (' + str(len(result)) + ')')
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
        logging.debug('Data loaded from path')
    else:
        logging.debug('Data loaded from memory')
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


# returns pair of images
# return ... [image1 or 2][number of pairs][image dimensions...], target values
# for 1 photo of person A
# select 1 photo of A from each camera
# select 1 photo of random person that is not A from each camera
def get_image_pairs(images, n_person_id, balance=True):
    # tuples of same people
    same = [[], []]
    # tuples of different people
    different = [[], []]

    info, unique_ids = _build_info(images, n_person_id)

    cameras = np.unique(info[:, 1])

    # for each person find him in each camera and find another person in each camera and make pairs
    for person_id in unique_ids:
        same_id = info[info[:, 0] == person_id]
        # get random image of person_id person
        person_image = same_id[random.randint(0, same_id.shape[0] - 1)][2]
        for camera in cameras:
            match = same_id[same_id[:, 1] == camera]
            if len(match) == 0:
                continue
            # get random image of person_id person from camera camera
            same_person_image = match[random.randint(0, match.shape[0] - 1)][2]
            same[0].append(person_image)
            same[1].append(same_person_image)
            diff = info[(info[:, 0] != person_id) & (info[:, 1] == camera)]
            if len(diff) == 0:
                continue
            # get random image of another person from camera camera
            different_person_image = diff[random.randint(0, diff.shape[0] - 1)][2]
            different[0].append(person_image)
            different[1].append(different_person_image)
    same, different = np.array(same), np.array(different)

    if balance:
        if same.shape[1] < different.shape[1]:
            different = different[:, :same.shape[1], :, :, :]
        elif same.shape[1] > different.shape[1]:
            same = same[:, :different.shape[1], :, :, :]
    inputs = np.append(same, different, axis=1)
    inputs = inputs / 255.0
    targets = np.concatenate((np.ones(same[0].shape[0]), np.zeros(different[0].shape[0])))
    return [inputs[0], inputs[1]], targets


# similar to get_image_pairs, but there is only one pair that is same and n_person_id-1 pairs that are not the same
# return ... [image1 or 2][number of pairs][image dimensions...], target values
# first image in the pair is SAME in all pairs
def get_oneshot_pair(images, n_person_id):
    info, unique_ids = _build_info(images, n_person_id)
    search_for = unique_ids[random.randint(0, unique_ids.shape[0] - 1)]
    # TODO: optimize to use numpy arrays
    images1 = []
    images2 = []
    targets = []

    same_id = info[info[:, 0] == search_for]
    person_image1 = same_id[random.randint(0, same_id.shape[0] - 1)][2] / 255.0

    # for each person find him in each camera and find another person in each camera and make pairs
    for person_id in unique_ids:
        same_id = info[info[:, 0] == person_id]
        # get random image of person_id person
        person_image2 = same_id[random.randint(0, same_id.shape[0] - 1)][2] / 255.0

        images1.append(person_image1)
        images2.append(person_image2)
        targets.append(int(person_id == search_for))

    return [np.array(images1), np.array(images2)], np.array(targets)


