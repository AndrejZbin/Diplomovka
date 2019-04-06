import os
import cv2
import re
import numpy as np

import logging
import random


# fetches image from location
# returns tuple containing loaded image and its file name
def fetch_images(path, file_type='.jpg'):
    result = []
    for file in os.listdir(path):
        if file.endswith(file_type) and os.path.isfile(os.path.join(path, file)):
            image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
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


def load_image(path, file):
    return cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR), file


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


# returns list of tuples containing 2 images of same person and list of tuples containing images of different people
# for 1 photo of person A
# select 1 photo of A from each camera
# select 1 photo of random person that is not A from each camera
def get_image_pairs(path, n_person_id):
    # tuples of same people
    same = []
    # tuples of different people
    different = []
    files = os.listdir(path)
    info = np.empty((len(files), 3), dtype=object)
    i = 0
    # info: person_id, camera_id, file_name
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            person_id, camera_id, _ = info_from_image_name(file)
            info[i] = np.array([person_id, camera_id, file],  dtype=object)
            i += 1
    # get n_person_id unique ids and ignore all the other people
    unique_ids = np.unique(info[:, 0])
    n_person_id = min(n_person_id, unique_ids.shape[0])
    unique_ids = unique_ids[:n_person_id]
    info = info[list(map(lambda el: el in unique_ids, info[:, 0])), :]

    cameras = np.unique(info[:, 1])

    # for each person find him in each camera and find another person in each camera and make pairs
    for person_id in unique_ids:
        same_id = info[info[:, 0] == person_id]
        person_image = load_image(path, same_id[random.randint(0, same_id.shape[0] - 1)][2])
        for camera in cameras:
            match = same_id[same_id[:, 1] == camera]
            if len(match) == 0:
                continue
            same_person_image = load_image(path, match[random.randint(0, match.shape[0] - 1)][2])
            same.append((person_image, same_person_image),)
            diff = info[(info[:, 0] != person_id) & (info[:, 1] == camera)]
            if len(diff) == 0:
                continue
            different_person_image = load_image(path, diff[random.randint(0, diff.shape[0] - 1)][2])
            different.append((person_image, different_person_image),)

    return np.array(same), np.array(different)


train_folder = os.path.join('DukeMTMC-reID', 'bounding_box_train')
test_folder = os.path.join('DukeMTMC-reID', 'bounding_box_test')
same, different = get_image_pairs(train_folder, 1)
print(same[0][0][1], same[0][1][1])
