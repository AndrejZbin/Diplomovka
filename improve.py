import os
import cv2
import logging
import numpy as np

import help_functions
import siamese_network
import config

from train_help_functions import get_image_pairs


# build a new dataset from information we gathered from this playback
def build_new_dataset(group_name, tracked_objects):
    # we save images here
    path = os.path.join(config.improve_folder, group_name)
    # create necessary folders
    if not os.path.exists(os.path.join(path, 'faces')):
        os.makedirs(os.path.join(path, 'faces'))
    if not os.path.exists(os.path.join(path, 'bodies')):
        os.makedirs(os.path.join(path, 'bodies'))
    # iterate over tracked people
    for track_id, track in tracked_objects.items():
        # if people were unmatched, they could cause false negatives, we don't want feed our model with that
        if not track.was_reided():
            continue
        faces = track.face_full_images
        faces_info = track.face_full_images_info

        bodies = track.body_full_images
        bodies_info = track.body_full_images_info
        for i in range(len(faces)):
            face = faces[i]
            frame, camera = faces_info[i]
            # make filename matching other dataset's filenames
            filename = str(track_id) + '_c' + str(camera) + '_f' + str(frame) + '.jpg'
            # finally save image
            cv2.imwrite(os.path.join(path, 'faces', filename), face)
        for i in range(len(bodies)):
            body = bodies[i]
            frame, camera = bodies_info[i]
            # make filename matching other dataset's filenames
            filename = str(track_id) + '_c' + str(camera) + '_f' + str(frame) + '.jpg'
            # finally save image
            cv2.imwrite(os.path.join(path, 'bodies', filename), body)

    fix_built_dataset(group_name)


# fix dataset by matching different IDs, check manually, also delete manually bad files
def fix_built_dataset(group_name):
    while True:
        r = input('Fix from to: ')
        r = r.split(' ')
        if len(r) != 2:
            break
        try:
            f, t = int(r[0]), int(r[1])
        except ValueError:
            break
        for path, _, files in os.walk(os.path.join(config.improve_folder, group_name)):
            for file in files:
                person_id, camera, frame = help_functions.info_from_image_name(file)
                if person_id == f:
                    filename = str(t) + '_c' + str(camera) + '_f' + str(frame) + '.jpg'
                    os.rename(os.path.join(path, file), os.path.join(path, filename))


# TODO: DRY, code is repeated for faces and bodies
# improve model for recognizing faces by learning from captured data
# potential problem A1 matched with A2, B1 matched with B2, but A1,B1,A2,B2 is one person
def improve_faces():
    # same person in different folders might have different ID, causing false negatives if we connected them
    train_images = []
    # load images from each folder
    for group_name in config.improve_camera_groups:
        images = help_functions.load_all_images(os.path.join(config.improve_folder, group_name, 'faces'),
                                                file_type='.jpg', preprocess=help_functions.prepare_face)
        train_images.append(images)

    # hyper-parameters
    people_count = 8
    iterations = 1000
    checkpoint = 20
    save_checkpoint = 5000

    model = siamese_network.get_face_model((config.face_image_resize[1], config.face_image_resize[0], 1))

    # are we improving base model of already improved model?
    # load weight for model we are improving
    if config.learning_start:
        model.load_weights(filepath=config.base_face_model)
    elif config.learning_improving:
        model.load_weights(filepath=config.improved_face_model)

    f = open(os.path.join('model_history', 'face_improve_perf.txt'), 'a')
    logging.info('IMPROVING: Starting to improve model for faces')
    for i in range(1, iterations+1):
        inputs, targets = get_image_pairs(train_images[np.random.randint(0, len(train_images))], people_count)
        (loss, acc) = model.train_on_batch(inputs, targets)
        if i % checkpoint == 0:
            logging.info('Iteration: {}'.format(i))
            logging.info('Loss: {}'.format(loss))
            logging.info('Accuracy: {}'.format(acc))
            f.write(str(i) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        if i % save_checkpoint == 0:
            model.save_weights(os.path.join('model_history', str(i) + 'FI.h5'))
            f.flush()
    model.save_weights(config.improved_face_model)
    f.close()


# improve model for recognizing bodies by learning from captured data
def improve_bodies():
    # same person in different folders might have different ID, causing false negatives if we connected them
    train_images = []
    # load images from each folder
    for group_name in config.improve_camera_groups:
        images = help_functions.load_all_images(os.path.join(config.improve_folder, group_name, 'bodies'),
                                                file_type='.jpg', preprocess=help_functions.prepare_body)
        train_images.append(images)

    # hyper-parameters
    people_count = 8
    iterations = 1000
    checkpoint = 20
    save_checkpoint = 5000

    model = siamese_network.get_body_model((config.body_image_resize[1], config.body_image_resize[0], 3))

    # are we improving base model of already improved model?
    # load weight for model we are improving
    if config.learning_start:
        model.load_weights(filepath=config.base_body_model)
    elif config.learning_improving:
        model.load_weights(filepath=config.improved_body_model)

    f = open(os.path.join('model_history', 'body_improve_perf.txt'), 'a')
    logging.info('IMPROVING: Starting to improve model for bodies')
    for i in range(1, iterations+1):
        inputs, targets = get_image_pairs(train_images[np.random.randint(0, len(train_images))], people_count)
        (loss, acc) = model.train_on_batch(inputs, targets)
        if i % checkpoint == 0:
            logging.info('Iteration: {}'.format(i))
            logging.info('Loss: {}'.format(loss))
            logging.info('Accuracy: {}'.format(acc))
            f.write(str(i) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        if i % save_checkpoint == 0:
            model.save_weights(os.path.join('model_history', str(i) + 'FBI.h5'))
            f.flush()
    model.save_weights(config.improved_body_model)
    f.close()


if __name__ == '__main__':
    improve_faces()
    improve_bodies()
