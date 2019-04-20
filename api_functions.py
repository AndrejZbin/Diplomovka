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

body_model = siamese_network.get_body_model((config.body_image_resize[1], config.body_image_resize[0], 3))
face_model = siamese_network.get_face_model((config.face_resize[1], config.face_resize[0], 1))

if config.learning_start:
    body_model.load_weights(filepath=config.base_body_mode)
    face_model.load_weights(filepath=config.base_face_model)
elif config.learning_improving or config.production:
    body_model.load_weights(filepath=config.improved_body_mode)
    face_model.load_weights(filepath=config.improved_face_model)

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')


def confirm_match(candidate_track, same_track):
    face1 = candidate_track.get_face_images()
    face2 = same_track.get_face_images()
    body1 = candidate_track.get_body_images()
    body1 = body1[np.random.randint(0, body1.shape[0])]
    body2 = same_track.get_body_images()
    body2 = body2[np.random.randint(0, body2.shape[0])]
    if face1.shape[0] > 0 and face2.shape[0] > 0:
        face1 = face1[np.random.randint(0, face1.shape[0])]
        face2 = face2[np.random.randint(0, face2.shape[0])]

        cv2.imshow('candidate face', face1.astype(np.uint8))
        cv2.imshow('matched to face', face2.astype(np.uint8))
    cv2.imshow('candidate', body1.astype(np.uint8))
    cv2.imshow('matched to', body2.astype(np.uint8))
    override = cv2.waitKey() & 0xFF == ord('n')
    cv2.destroyWindow('candidate')
    cv2.destroyWindow('matched to')
    cv2.destroyWindow('candidate face')
    cv2.destroyWindow('matched to face')
    return not override


def prepare_face(image):
    # image = retinex.automatedMSRCR(
    #     image,
    #     [15, 80, 250]
    # )
    image = cv2.resize(image, config.face_resize)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    return image


def prepare_body(image):
    # image = retinex.automatedMSRCR(
    #     image,
    #     [15, 80, 250]
    # )
    image = cv2.resize(image, config.body_image_resize)
    return image


def detect_people(image):
    # rectangles, weights = descriptor.detectMultiScale(
    #     image, winStride=(4, 4), padding=(0, 0), scale=1.05, hitThreshold=1)
    # rectangles = np.array([[x + 0.05*w, y + 0.05*h, x + 0.95*w, y + 0.95*h] for x, y, w, h in rectangles])
    # rectangles = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)
    # return rectangles
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
            scaleFactor=1.02, minNeighbors=5, minSize=(30, 30))
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
        return np.array([])
    match = body_model.predict_on_batch(_build_compare(image/255.0, images/255.0))
    return match


def compare_bodies_to_bodies(bodies1, bodies2):
    match = []
    bodies1 = bodies1 / 255.0
    bodies2 = bodies2 / 255.0
    if len(bodies1) == 0 or len(bodies2) == 0:
        return np.array([[]])
    for body in bodies1:
        match.append(body_model.predict_on_batch(_build_compare(body, bodies2)))
    return np.array(match)


def compare_face(image1, image2):
    return compare_face_to_faces(image1, np.array([image2]))[0]


def compare_face_to_faces(image, images):
    if len(images) == 0:
        return np.array([])
    match = face_model.predict_on_batch(_build_compare(image/255.0, images/255.0))
    return match


def compare_faces_to_faces(faces1, faces2):
    match = []
    faces1 = faces1 / 255.0
    faces2 = faces2 / 255.0
    if len(faces1) == 0 or len(faces2) == 0:
        return np.array([[]])
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


def _match_face(match):
    if not config.avoid_false_positives:
        return match
    if match.size == 1:
        return match >= 0.95
    elif match.size == 2:
        return match >= 0.90
    elif match.size == 3:
        return match >= 0.75
    elif match.size == 4:
        return match >= 0.65
    return match >= 0.60


def _match_body(match):
    if not config.avoid_false_positives:
        return match
    if match.size == 1:
        return match >= 0.95
    elif match.size == 2:
        return match >= 0.92
    elif match.size == 3:
        return match >= 0.90
    elif match.size == 4:
        return match >= 0.88
    return match >= 0.80


def compare_to_detected(candidate_track, saved_objects):

    # for each saved object get all samples and compare them to candidate samples and average results,
    # then select the largest average match and return id of that object

    # try to match by face
    match_face = np.array([
        np.mean(_match_face(compare_faces_to_faces(candidate_track.get_face_samples(), person_track.get_face_samples())))
        if person_id != candidate_track.get_id() else 0 for person_id, person_track in saved_objects.items()])
    match_body = np.array([
        np.mean(_match_body(compare_bodies_to_bodies(candidate_track.get_body_samples(), person_track.get_body_samples())))
        if person_id != candidate_track.get_id() else 0 for person_id, person_track in saved_objects.items()])

    match = np.empty(match_face.shape[0])
    for i in range(match.shape[0]):
        if np.isnan(match_face[i]) and np.isnan(match_body[i]):
            match[i] = 0
        elif np.isnan(match_face[i]):
            match[i] = match_body[i]
        elif np.isnan(match_body[i]):
            match[i] = match_face[i]
        else:
            dif = abs(match_face[i] - match_body[i])
            v = min(match_face[i], match_body[i]) + 0.75*dif
            match[i] = v

    if len(match) > 0:
        if config.confirm_match and config.confirm_match_count != 1:
            if config.confirm_match_count > 1:
                largest_indices = np.argsort(-match)[:config.confirm_match_count]
            else:
                largest_indices = (match >= config.required_match_count).nonzero()[0]
        else:
            largest_indices = [int(np.argmax(match))]
        for largest_index in largest_indices:
            same_id, same_track = list(saved_objects.items())[largest_index]
            candidate_id = candidate_track.get_id()
            if same_id == candidate_id:  # match is 0
                continue
            match_count = int(match[largest_index]*10000)/100.0
            if match[largest_index] >= config.required_match_count or config.confirm_match:
                print('Matched {} to {} match {}%'.format(candidate_id, same_id, match_count))
                if config.confirm_match:
                    if confirm_match(candidate_track, same_track):
                        print('MATCH CONFIRMED')
                        return same_id
                    else:
                        print('MATCH OVERRIDDEN')
                else:
                    return same_id
            else:
                print('NOT Matched {} to {} match {}%'.format(candidate_id, same_id, match_count))

    return None


