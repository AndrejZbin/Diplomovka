import numpy as np

import cv2

from imutils.object_detection import non_max_suppression

descriptor = cv2.HOGDescriptor()
descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_body(image):
    rectangles, weights = descriptor.detectMultiScale(
        image, winStride=(4, 4), padding=(0, 0), scale=1.05, hitThreshold=1)
    rectangles = np.array([[x + 0.05*w, y + 0.05*h, x + 0.95*w, y + 0.95*h] for x, y, w, h in rectangles])
    rectangles = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)
    return rectangles


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
