import numpy as np
import sys

import config
import help_functions
import train_help_functions
from siamese_network import get_body_model, get_face_model


def test_body(model_file):
    test_images = help_functions.load_all_images(
        config.test_body_folder, preprocess=help_functions.prepare_body)
    model = get_body_model((config.body_image_resize[1], config.body_image_resize[0], 3))
    model.load_weights(filepath=model_file)

    print('body')
    for i in range(10):
        inputs, targets = train_help_functions.get_image_pairs(test_images, 10)

        result = model.test_on_batch(inputs, targets)
        print(result)


def test_body_oneshot(model_file, iterations=10, versus=4):
    test_images = help_functions.load_all_images(
        config.test_body_folder, preprocess=help_functions.prepare_body)
    model = get_body_model((config.body_image_resize[1], config.body_image_resize[0], 3))
    model.load_weights(filepath=model_file)

    matched = 0
    for i in range(iterations):
        inputs, targets = train_help_functions.get_oneshot_pair(test_images, versus)
        result = model.predict_on_batch(inputs)
        matched += np.argmax(result) == np.argmax(targets)
    print('Oneshot body:', float(matched)/float(iterations), 'vs', versus)


def test_face(model_file):
    test_images = help_functions.load_all_images(
        config.chokepoint_cropped_test, file_type='.pgm', preprocess=help_functions.prepare_face)
    model = get_face_model((config.face_image_resize[1], config.face_image_resize[0], 1))
    model.load_weights(filepath=model_file)
    print('face')
    for i in range(10):
        inputs, targets = train_help_functions.get_image_pairs(test_images, 10)

        result = model.test_on_batch(inputs, targets)
        print(result)


def test_face_oneshot(model_file, iterations=10, versus=4):
    test_images = help_functions.load_all_images(
        config.chokepoint_cropped_test, file_type='.pgm', preprocess=help_functions.prepare_face)
    model = get_face_model((config.face_image_resize[1], config.face_image_resize[0], 1))
    model.load_weights(filepath=model_file)

    matched = 0
    for i in range(iterations):
        inputs, targets = train_help_functions.get_oneshot_pair(test_images, versus)
        result = model.predict_on_batch(inputs)
        matched += np.argmax(result) == np.argmax(targets)
    print('Oneshot face:', float(matched)/float(iterations), 'vs', versus)


if __name__ == '__main__':
    vs = 16
    if len(sys.argv) == 2:
        vs = int(sys.argv[1])
    test_face(config.base_face_model)
    test_face_oneshot(config.base_face_model, 100, vs)

    test_body(config.base_body_model)
    test_body_oneshot(config.base_body_model, 100, vs)
