import cv2
import os

import feature_extractors as fe

import dlib
from centroid_tracker import CentroidTracker
from person_track import PersonTrack

import api_functions

from playback import *

import config

cams = ['P2E_S4_C1.1', 'P2E_S4_C2.1', 'P2E_S3_C3.1']
imgListPaths = list(map(lambda c: os.path.join(c, 'all_file.txt'), cams))
print(imgListPaths)

players = list(map(lambda l: PicturePlayback([os.path.join(os.path.dirname(l), line.rstrip('\n')) for line in open(l)], 30, True), imgListPaths))
#players = [CameraPlayback()]
# player = [VideoPlayback('video.avi')]
#players = [YoutubePlayback('https://www.youtube.com/watch?v=N79f1znMWQ8')]

cts = [CentroidTracker(0, 100) for _ in range(len(cams))]
trackerss = [[] for _ in range(len(cams))]
trackableObjects = {}
trackableObjectsReID = {}


face_cascade = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv_data/haarcascades/haarcascade_eye.xml')

for player in players:
    player.start()


frame_index = 0
next_id = 1
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
        faces = []
        objects = None
        if frame_index % config.detect_frequency == 0:
            print('Detecting now')
            trackerss[camera_i] = []
            people = api_functions.detect_body(frame)
            for (x1, y1, x2, y2) in people:
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x1, y1, x2, y2)
                tracker.start_track(frame, rect)
                trackerss[camera_i].append(tracker)
            objects = cts[camera_i].update(people, True)
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
            objects = cts[camera_i].update(people, False)
        # loop over the tracked objects
        for (objectID, (x1, y1, x2, y2)) in objects.items():
            x2, x1, y2, y1 = min(x2, frame.shape[1]), max(x1, 0), min(y2, frame.shape[1]), max(y1, 0)
            if x2-x1 < 20 or y2 - y1 < 40:
                continue
            # check to see if a trackable object exists for the current
            # object ID
            objectID = trackableObjectsReID.get(objectID, objectID)
            to = trackableObjects.get(objectID, None)

            cropped_body = frame[y1:y2, x1:x2]

            # if there is no existing trackable object, create one
            if to is None:
                same_person_id = api_functions.compare_to_detected(objectID, cropped_body, trackableObjects)
                same_person_id = trackableObjectsReID.get(same_person_id, same_person_id)
                if same_person_id is None:
                    to = PersonTrack(objectID, len(cams))
                    trackableObjects[objectID] = to
                else:
                    trackableObjectsReID[objectID] = same_person_id
                    print(objectID, same_person_id)
                    objectID = same_person_id
                    to = trackableObjects.get(same_person_id)
            elif frame_index % (config.detect_frequency * 2) == 0:
                same_person_id = api_functions.compare_to_detected(objectID, cropped_body, trackableObjects)
                same_person_id = trackableObjectsReID.get(same_person_id, same_person_id)
                if same_person_id is not None and same_person_id != objectID:
                    del trackableObjects[objectID]
                    trackableObjectsReID[objectID] = same_person_id
                    print(objectID, same_person_id)
                    objectID = same_person_id
                    to = trackableObjects.get(same_person_id)


            if frame_index % config.detect_frequency == 0:
                to.add_body_sample(cropped_body, camera_i)
                face = api_functions.detect_face(cropped_body)
                if len(face) == 1:
                    fx1, fy1, fx2, fy2 = face[0]
                    to.add_face_sample(cropped_body[fy1:fy2, fx1:fx2], camera_i)

            trackableObjects[objectID] = to
            cv2.putText(frame, to.get_name(), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frames[camera_i], (x1, y1), (x2, y2), (0, 255, 0), 2)










    # for camera_i, frame in enumerate(frames):
    #     people = []
    #     if frame_index % config.detect_frequency == 0:
    #         print('Detecting now')
    #         trackerss[camera_i] = []
    #         rectangles = api_functions.detect_body(frame)
    #         for x1, y1, x2, y2 in rectangles:
    #             x2, x1, y2, y1 = min(x2, frame.shape[1]), max(x1, 0), min(y2, frame.shape[1]), max(y1, 0)
    #             if x2 - x1 < 20 or y2 - y1 < 40:
    #                 continue
    #
    #             cropped_body = frame[y1:y2, x1:x2]
    #             same_person_id = api_functions.compare_to_detected(cropped_body, trackableObjects)
    #
    #             if same_person_id is None:
    #                 same_person_id = next_id
    #                 trackableObjects[same_person_id] = PersonTrack(same_person_id, len(cams))
    #                 next_id += 1
    #
    #             person = trackableObjects.get(same_person_id)
    #             person.add_body_sample(cropped_body, camera_i)
    #
    #             tracker = dlib.correlation_tracker()
    #             rect = dlib.rectangle(x1, y1, x2, y2)
    #             tracker.start_track(frame, rect)
    #             trackerss[camera_i].append((same_person_id, tracker),)
    #
    #             people.append((same_person_id, (x1, y1, x2, y2)), )
    #     else:
    #         for person_id, tracker in trackerss[camera_i]:
    #             # update the tracker and grab the updated position
    #             tracker.update(frame)
    #             pos = tracker.get_position()
    #
    #             # unpack the position object
    #             x1 = int(pos.left())
    #             y1 = int(pos.top())
    #             x2 = int(pos.right())
    #             y2 = int(pos.bottom())
    #
    #             # add the bounding box coordinates to the rectangles list
    #             people.append((person_id, (x1, y1, x2, y2)),)
    #
    #     # loop over the tracked objects
    #     for (objectID, (x1, y1, x2, y2)) in people:
    #         x2, x1, y2, y1 = min(x2, frame.shape[1]), max(x1, 0), min(y2, frame.shape[1]), max(y1, 0)
    #         if x2-x1 < 20 or y2 - y1 < 40:
    #             continue
    #         # check to see if a trackable object exists for the current
    #         # object ID
    #         to = trackableObjects.get(objectID, None)
    #         cv2.putText(frame, to.get_name(), (x1, y1),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #         cv2.rectangle(frames[camera_i], (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
        cv2.imshow('Camera ' + str(camera_i), frames[camera_i])



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1


cv2.destroyAllWindows()
