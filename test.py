import numpy as np
import sys

import config
import help_functions
import train_help_functions
from siamese_network import get_body_model, get_face_model


def calc_rates(predicted, truth):
    truth = np.squeeze(truth.astype(bool))
    predicted = np.squeeze(predicted >= 0.5)
    tp = np.count_nonzero(truth & predicted)
    tn = np.count_nonzero(np.logical_not(truth) & np.logical_not(predicted))
    fp = np.count_nonzero(np.logical_not(truth) & predicted)
    fn = np.count_nonzero(truth & np.logical_not(predicted))
    return np.array([tp, tn, fp, fn])


def print_rates(rates):
    tp = rates[0]
    tn = rates[1]
    fp = rates[2]
    fn = rates[3]
    p = tp + fn
    n = tn + fp
    acc = (tp + tn) / (p + n)
    tpr = tp / p
    tnr = tn / n
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    print('True positive: {}'.format(tp))
    print('True negative: {}'.format(tn))
    print('False positive: {}'.format(fp))
    print('False negative: {}'.format(fn))
    print('Accuracy: {}'.format(acc))
    print('True positive rate: {}'.format(tpr))
    print('True negative rate: {}'.format(tnr))
    print('Precision: {}'.format(ppv))
    print('Negative predictive value: {}'.format(npv))


def test_body(model_file):
    test_images = help_functions.load_all_images(
        config.test_body_folder, preprocess=help_functions.prepare_body)
    model = get_body_model((config.body_image_resize[1], config.body_image_resize[0], 3))
    model.load_weights(filepath=model_file)
    print('body')
    rates = np.array([0, 0, 0, 0])
    for i in range(100):
        inputs, targets = train_help_functions.get_image_pairs(test_images, 10)
        predicted = model.predict_on_batch(inputs)
        rates += calc_rates(predicted, targets)
    print_rates(rates)


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
    # print(versus, float(matched)/float(iterations))


def test_face(model_file):
    test_images = help_functions.load_all_images(
        config.chokepoint_cropped_test, file_type='.pgm', preprocess=help_functions.prepare_face)
    model = get_face_model((config.face_image_resize[1], config.face_image_resize[0], 1))
    model.load_weights(filepath=model_file)
    print('face')
    rates = np.array([0, 0, 0, 0])
    for i in range(100):
        inputs, targets = train_help_functions.get_image_pairs(test_images, 10)
        predicted = model.predict_on_batch(inputs)
        rates += calc_rates(predicted, targets)
    print_rates(rates)


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
    # print(versus, float(matched) / float(iterations))


if __name__ == '__main__':
    vs = 16
    if len(sys.argv) == 2:
        vs = int(sys.argv[1])
    test_face(config.base_face_model)
    test_face_oneshot(config.base_face_model, 100, vs)

    test_body(config.base_body_model)
    test_body_oneshot(config.base_body_model, 100, vs)

    # for i in range(1, 26):
    #     test_face_oneshot(config.base_face_model, 100, i)
    # for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     test_body_oneshot(config.base_body_model, 100, i)
