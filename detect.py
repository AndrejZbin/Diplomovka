import cv2
import numpy as np

from imutils.object_detection import non_max_suppression

import config

body_descriptor = cv2.HOGDescriptor()
body_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

face_descriptor = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromCaffe(config.mobilenet_proto, config.mobilenet_model)

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'dining_table', 'dog', 'horse', 'motorbike', 'person', 'potted_plant', 'sheep', 'sofa', 'train', 'tv_monitor'
           )


def detect_people(image):
    if config.body_hog:
        rectangles, weights = body_descriptor.detectMultiScale(
            image, winStride=(4, 4), padding=(0, 0), scale=1.05, hitThreshold=1)
        rectangles = np.array([[x + 0.05*w, y + 0.05*h, x + 0.95*w, y + 0.95*h] for x, y, w, h in rectangles])
        rectangles = non_max_suppression(rectangles, probs=None, overlapThresh=0.65)
        return rectangles
    else:
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

    faces = face_descriptor.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=5, minSize=(30, 30))
    faces = list(map(lambda t: (t[0], t[1], t[0]+t[2], t[1]+t[3]), faces))
    return faces


def get_face(image):
    faces = detect_faces(image)
    if len(faces) == 1:
        fx1, fy1, fx2, fy2 = faces[0]
        return image[fy1:fy2, fx1:fx2]
    return None
