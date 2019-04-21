import numpy as np

import logging
import random

import config

from help_functions import _build_info


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

    if different.shape == 0:
        logging.error('BUILDING PAIRS: cannot create pairs because there is 0 different people')
        raise FileNotFoundError()

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
    # TODO: maybe optimize to use numpy arrays
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


def get_image_pair_batch(people_count=10, folder=config.train_body_folder):
    while True:
        inputs, targets = get_image_pairs(folder, people_count)
        yield inputs, targets
