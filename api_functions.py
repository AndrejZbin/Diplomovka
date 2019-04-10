import numpy as np


def detect_body(image):
    pass


def detect_face(image):
    pass


def compare_body(image1, image2):
    return compare_body_to_bodies(image1, [image2])[0]


def compare_body_to_bodies(image, images, names=None):
    match = []
    if names is None:
        return match
    return get_name(match, names)


def compare_face(image1, image2):
    return compare_face_to_faces(image1, [image2])[0]


def compare_face_to_faces(image, images, names=None):
    match = []
    if names is None:
        return match
    return get_name(match, names)


def _build_compare(image, images):
    return [[image, i] for i in images]


def get_name(match, names):
    return names[np.argmax(match)]
