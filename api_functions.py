import numpy as np

import cv2

from imutils.object_detection import non_max_suppression

import siamese_network

import config
import retinex

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


def prepare_face(image):
    # image = retinex.automatedMSRCR(
    #     image,
    #     [15, 80, 250]
    # )
    image = cv2.resize(image, config.face_resize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    return image


def prepare_body(image):
    image = retinex.automatedMSRCR(
        image,
        [15, 80, 250]
    )
    image = cv2.resize(image, config.body_image_resize)
    return image


def detect_people(image):
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
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= config.detect_body_confidence:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != 'person':
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype("int")
            rectangles.append((x1, y1, x2, y2),)
    return rectangles


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_descriptor.detectMultiScale(gray,
            scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    faces = list(map(lambda t: (t[0], t[1], t[0]+t[2], t[1]+t[3]), faces))
    return faces


def get_face(image):
    faces = detect_faces(image)
    if len(faces) == 1:
        fx1, fy1, fx2, fy2 = faces[0]
        return image[fy1:fy2, fx1:fx2]
    return None


def compare_body_to_body(image1, image2):
    return compare_body_to_bodies(image1, np.array([image2]))[0]


def compare_body_to_bodies(image, images):
    if len(images) == 0:
        return np.array([0])
    match = body_model.predict_on_batch(_build_compare(image/255.0, images/255.0))
    return match


def compare_bodies_to_bodies(bodies1, bodies2):
    match = []
    bodies1 = bodies1 / 255.0
    bodies2 = bodies2 / 255.0
    if len(bodies1) == 0 or len(bodies2) == 0:
        return np.array([[0]])
    for body in bodies1:
        match.append(body_model.predict_on_batch(_build_compare(body, bodies2)))
    return np.array(match)


def compare_face(image1, image2):
    return compare_face_to_faces(image1, np.array([image2]))[0]


def compare_face_to_faces(image, images):
    if len(images) == 0:
        return np.array([0])
    match = face_model.predict_on_batch(_build_compare(image/255.0, images/255.0))
    return match


def compare_faces_to_faces(faces1, faces2):
    match = []
    faces1 = faces1 / 255.0
    faces2 = faces2 / 255.0
    if len(faces1) == 0 or len(faces2) == 0:
        return np.array([[0]])
    for face in faces1:
        match.append(face_model.predict_on_batch(_build_compare(face, faces2)))
    return np.array(match)


def _build_compare(image, images):
    a1 = []
    for _ in range(len(images)):
        a1.append(image)
    return [a1, images]


def get_name(match, names):
    if not names or len(names) == 0:
        return None, 0
    return names[np.argmax(match)], np.max(match)


def compare_to_detected(candidate_track, saved_objects):

    # for each saved object get all samples and compare them to candidate samples and average results,
    # then select the largest average match and return id of that object

    # try to match by face
    match_face = np.array([
        np.max(
            np.mean(np.mean(compare_faces_to_faces(candidate_track.get_face_samples(), person_track.get_face_samples()), axis=1))
        )
        for person_id, person_track in saved_objects.items() if person_id != candidate_track.get_id()])
    if len(match_face) > 0:
        largest_index = int(np.argmax(match_face))
        if match_face[largest_index] >= 0.90:
            print('Matched by FACE ' + str(int(match_face[largest_index]*10000)/100.0))
            return list(filter(lambda k: k != candidate_track.get_id(), saved_objects.keys()))[largest_index]

    match_body = np.array([
        np.max(
            np.mean(np.mean(compare_bodies_to_bodies(candidate_track.get_body_samples(), person_track.get_body_samples()), axis=1))
        )
        for person_id, person_track in saved_objects.items() if person_id != candidate_track.get_id()
    ])

    if len(match_body) == 0:
        return None
    largest_index = int(np.argmax(match_body))
    if match_body[largest_index] >= 0.90:
        print('Matched by BODY ' + str(int(match_body[largest_index]*10000)/100.0))
        return list(filter(lambda k: k != candidate_track.get_id(), saved_objects.keys()))[largest_index]
    return None  # no match found



