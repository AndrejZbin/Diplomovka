import logging
import os

from keras import backend

import config
import help_functions

from train_help_functions import get_image_pairs
from siamese_network import get_body_model, get_face_model


def train_body():
    train_images = help_functions.load_all_images(
        config.train_body_folder, preprocess=help_functions.prepare_body)

    # Hyper-parameters
    people_count = 8
    iterations = 60000
    checkpoint = 20
    save_checkpoint = 10000

    backend.clear_session()
    model = get_body_model((config.body_image_resize[1], config.body_image_resize[0], 3))
    f = open(os.path.join('model_history', 'body_perf.txt'), 'a')
    # print(model.summary())
    for i in range(1, iterations+1):
        inputs, targets = get_image_pairs(train_images, people_count)
        (loss, acc) = model.train_on_batch(inputs, targets)

        if i % checkpoint == 0:
            logging.info('Iteration: {}'.format(i))
            logging.info('Loss: {}'.format(loss))
            logging.info('Accuracy: {}'.format(acc))
            f.write(str(i) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        if i % save_checkpoint == 0:
            model.save_weights(os.path.join('model_history', 'base_body_weights_it{}.h5'.format(i)))
    model.save_weights(os.path.join('model_history', 'base_body_weights.h5'))
    f.close()


def train_face():
    train_images = help_functions.load_all_images(
        config.chokepoint_cropped_train, file_type='.pgm', preprocess=help_functions.prepare_face_random_resize)

    # Hyper-parameters
    people_count = 8
    iterations = 25000
    checkpoint = 20
    save_checkpoint = 10000

    backend.clear_session()
    model = get_face_model((config.face_image_resize[1], config.face_image_resize[0], 1))
    f = open(os.path.join('model_history', 'face_perf.txt'), 'a')
    # print(model.summary())
    for i in range(1, iterations+1):
        inputs, targets = get_image_pairs(train_images, people_count)
        (loss, acc) = model.train_on_batch(inputs, targets)
        if i % checkpoint == 0:
            logging.info('Iteration: {}'.format(i))
            logging.info('Loss: {}'.format(loss))
            logging.info('Accuracy: {}'.format(acc))
            f.write(str(i) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        if i % save_checkpoint == 0:
            model.save_weights(os.path.join('model_history', 'base_face_weights_it{}.h5'.format(i)))
            f.flush()
    model.save_weights(os.path.join('model_history', 'base_face_weights.h5'))
    f.close()


if __name__ == '__main__':
    backend.clear_session()
    train_face()
    train_body()
