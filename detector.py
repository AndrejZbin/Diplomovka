import cv2
import os

import feature_extractors as fe

import dlib
from centroid_tracker import CentroidTracker
from person_track import PersonTrack

import api_functions

from playback import *

cams = ['P2E_S4_C1.1']
imgListPaths = list(map(lambda c: os.path.join(c, 'all_file.txt'), cams))
print(imgListPaths)

players = list(map(lambda l: PicturePlayback([os.path.join(os.path.dirname(l), line.rstrip('\n')) for line in open(l)], 30, True), imgListPaths))
#players = [CameraPlayback()]
# player = [VideoPlayback('video.avi')]
#players = [YoutubePlayback('https://www.youtube.com/watch?v=3aADeK-bSMU')]

cts = [CentroidTracker(60, 50) for _ in range(len(cams))]
trackerss = [[] for _ in range(len(cams))]
trackableObjects = {}


face_cascade = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_eye.xml')

for player in players:
    player.start()


frame_index = 0
while all([player.is_playing() for player in players]):
    frames = [player.get_frame() for player in players]
    if any(list(map(lambda frame: frame is None, frames))):
        print('Ended')
        break

    grays = list(map(lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), frames))



    facess = list(map(lambda gray: face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    ), grays))

    frame_copies = [frame.copy() for frame in frames]


    # # Draw a rectangle around the faces
    # for camera_i, faces in enumerate(facess):
    #     for face_i, (x, y, w, h) in enumerate(faces):
    #         roi_gray = grays[camera_i][y:y + h, x:x + w]
    #         roi_color = frames[camera_i][y:y + h, x:x + w]
    #         eyes = eye_cascade.detectMultiScale(roi_gray)
    #         for (ex, ey, ew, eh) in eyes:
    #             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
    #
    #         if len(faces) == 1 and len(eyes) == 2:
    #             cv2.imshow('face camera ' + str(camera_i),  cv2.resize(frame_copies[camera_i][y:y + h, x:x + w], (256, 256)))
    #
    #         cv2.rectangle(frames[camera_i], (x, y), (x+w, y+h), (0, 255, 0), 2)
    #
    #     cv2.imshow('Camera ' + str(camera_i), frames[camera_i])
    for camera_i, frame in enumerate(frames):
        people = []
        if frame_index % 30 == 0:
            trackerss[camera_i] = []
            people = api_functions.detect_body(frame)
            for x1, y1, x2, y2 in people:
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x1, y1, x2, y2)
                tracker.start_track(frame, rect)
                trackerss[camera_i].append(tracker)
        else:
            for tracker in trackerss[camera_i]:
                # update the tracker and grab the updated position
                tracker.update(frame)
                pos = tracker.get_position()

                # unpack the position object
                x1 = int(pos.left())
                y1 = int(pos.top())
                x2 = int(pos.right())
                y2 = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                people.append((x1, y1, x2, y2))

        objects = cts[camera_i].update(people)
        # loop over the tracked objects
        for (objectID, (x1, y1, x2, y2)) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = PersonTrack(objectID, len(cams))

            trackableObjects[objectID] = to
            cv2.putText(frame, to.get_name(), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frames[camera_i], (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Camera ' + str(camera_i), frames[camera_i])



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1


cv2.destroyAllWindows()
