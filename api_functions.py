import numpy as np

import cv2

from imutils.object_detection import non_max_suppression

import siamese_network

import config

descriptor = cv2.HOGDescriptor()
descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

face_descriptor = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_frontalface_default.xml')

net = cv2.dnn.readNetFromCaffe(config.net_proto, config.net_model)

body_model = siamese_network.get_model((config.body_image_resize[1], config.body_image_resize[0], 3))
body_model.load_weights(filepath=config.body_model)

face_model = siamese_network.get_model((config.face_resize[1], config.face_resize[0], 1))
face_model.load_weights(filepath=config.face_model)

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')


def detect_body(image):
    # rectangles, weights = descriptor.detectMultiScale(
    #     image, winStride=(4, 4), padding=(0, 0), scale=1.05, hitThreshold=1)
    # rectangles = np.array([[x + 0.05*w, y + 0.05*h, x + 0.95*w, y + 0.95*h] for x, y, w, h in rectangles])
    # rectangles = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)
    (H, W) = image.shape[0], image.shape[1]
    image = image.astype(np.float32)
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    rectangles = []
    faces = []
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by requiring a minimum
        # confidence
        if confidence >= 0.5:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])

            # if the class label is not a person, ignore it
            if CLASSES[idx] != 'person':
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype("int")
            rectangles.append((x1, y1, x2, y2),)
    return rectangles


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_descriptor.detectMultiScale(gray,
            scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    faces = list(map(lambda t: (t[0], t[1], t[0]+t[2], t[1]+t[3]), faces))
    return faces


def compare_body(image1, image2):
    return compare_body_to_bodies(image1, np.array([image2]))[0]


def compare_body_to_bodies(image, images, names=None):
    if len(images) == 0:
        return [0]
    match = body_model.predict_on_batch(_build_compare(image/255.0, images/255.0))
    if names is None:
        return match
    return get_name(match, names)


def compare_face(image1, image2):
    return compare_face_to_faces(image1, np.array([image2]))[0]


def compare_face_to_faces(image, images, names=None):
    if len(images) == 0:
        return [0]
    match = face_model.predict_on_batch(_build_compare(image/255.0, images/255.0))
    if names is None:
        return match
    return get_name(match, names)


def _build_compare(image, images):
    a1 = []
    for _ in range(len(images)):
        a1.append(image)
    return [a1, images]


def get_name(match, names):
    return names[np.argmax(match)]


def compare_to_detected(my_id, cropped_body, trackableObjects):
    import retinex
    cropped_body = retinex.automatedMSRCR(
        cropped_body,
        [15, 80, 250]
    )

    face = detect_face(cropped_body)
    if len(face) == 1:
        fx1, fy1, fx2, fy2 = face[0]
        face = cv2.cvtColor(cropped_body[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, config.face_resize)
        face = np.expand_dims(face, axis=2)
    else:
        face = None

    cropped_body = cv2.resize(cropped_body, config.body_image_resize)

    # try to match by face
    if face is not None:
        match_face = np.array([np.mean(compare_face_to_faces(face, person.get_face_samples())) for key, person in trackableObjects.items() if key != my_id])
        cv2.imshow('Camera', face)
        cv2.waitKey()
        if len(match_face) > 0:
            largest_index = np.argmax(match_face)
            if match_face[largest_index] >= 0.9:
                print('Matched by FACE')
                return list(trackableObjects.keys())[largest_index]

    # try to match by body

    # for each object get all samples and compare them to cropped_body and average results, then select the largest average match and return id of that object
    match = np.array([np.mean(compare_body_to_bodies(cropped_body, person.get_body_samples())) for key, person in trackableObjects.items() if key != my_id])

    if len(match) == 0:
        return None
    largest_index = np.argmax(match)
    if match[largest_index] >= 0.9:
        print('Matched by BODY')
        return list(trackableObjects.keys())[largest_index]
    return None  # no match found



