import os
import cv2
import re
import numpy as np

import logging
import random

from config import test_folder, train_folder, body_image_resize


# fetches image from location
# returns tuple containing loaded image and its file name
def fetch_images(path, file_type='.jpg'):
    result = []
    for file in os.listdir(path):
        if file.endswith(file_type) and os.path.isfile(os.path.join(path, file)):
            image, _ = load_image(path, file)
            result.append((image, file),)
    return result


saved_images = {}


# returns list of tuples containing loaded images and their file name
def load_all_images(path, file_type='.jpg'):
    key = path + '#' + file_type
    images = saved_images.get(key)
    if images is None:
        images = fetch_images(path, file_type)
        saved_images[key] = images
        logging.debug('Images loaded from path')
    else:
        logging.debug('Images loaded from memory')
    return images


def load_image(path, file, resize=body_image_resize):
    image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
    if resize is not None:
        image = cv2.resize(image, resize)
    return image, file


# files <- iterable, contains names of images to load
def load_images(path, files):
    result = []
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            image = load_image(path, file)
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


# returns tuple containing pair of images, first element are same, second different people
# for 1 photo of person A
# select 1 photo of A from each camera
# select 1 photo of random person that is not A from each camera
def get_image_pairs(path, n_person_id):
    # tuples of same people
    same = [[], []]
    # tuples of different people
    different = [[], []]

    images = load_all_images(path)

    files = os.listdir(path)
    info = np.empty((len(files), 3), dtype=object)
    i = 0
    # info: person_id, camera_id, file_name
    for image, file in images:
            person_id, camera_id, _ = info_from_image_name(file)
            info[i] = np.array([person_id, camera_id, image],  dtype=object)
            i += 1
    # info = info[:i, :]
    # get n_person_id unique ids and ignore all the other people
    unique_ids = np.unique(info[:, 0])
    np.random.shuffle(unique_ids)
    n_person_id = min(n_person_id, unique_ids.shape[0])
    unique_ids = unique_ids[:n_person_id]
    info = info[list(map(lambda el: el in unique_ids, info[:, 0])), :]

    cameras = np.unique(info[:, 1])

    # for each person find him in each camera and find another person in each camera and make pairs
    for person_id in unique_ids:
        same_id = info[info[:, 0] == person_id]
        person_image = same_id[random.randint(0, same_id.shape[0] - 1)][2]
        for camera in cameras:
            match = same_id[same_id[:, 1] == camera]
            if len(match) == 0:
                continue
            same_person_image = match[random.randint(0, match.shape[0] - 1)][2]
            same[0].append(person_image)
            same[1].append(same_person_image)
            diff = info[(info[:, 0] != person_id) & (info[:, 1] == camera)]
            if len(diff) == 0:
                continue
            different_person_image = diff[random.randint(0, diff.shape[0] - 1)][2]
            different[0].append(person_image)
            different[1].append(different_person_image)
    return np.array(same), np.array(different)


# [picture1 or 2][pair index][dimensions of picture]
def pairs_prepare(same, different, balance=True):
    # connect same and different
    if balance:
        if same.shape[1] < different.shape[1]:
            different = different[:, :same.shape[1], :, :, :]
        elif same.shape[1] > different.shape[1]:
            same = same[:, :different.shape[1], :, :, :]
    inputs = np.append(same, different, axis=1)
    inputs = inputs / 255.0
    targets = np.concatenate((np.ones(same[0].shape[0]), np.zeros(different[0].shape[0])))
    return inputs, targets

