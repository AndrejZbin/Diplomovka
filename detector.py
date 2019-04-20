import cv2
import os

import feature_extractors as fe

import help_functions

import re
import math

import dlib
from centroid_tracker import CentroidTracker
from person_track import PersonTrack

import api_functions

from playback import *

import config

import numpy as np

# saved information about detected people, key=ID, value=person_track instance
tracked_objects = {}
# id from tracker -> id of person
tracked_objects_reid = {}

cams = ['P2E_S4_C1.1', 'P2E_S4_C2.1', 'P2E_S3_C3.1']


def build_known_people():
    images = help_functions.load_all_images(config.keep_track_targeted_files, '', help_functions.resize_face)
    name_to_id = {}
    next_id = -1
    cameras = len(cams)
    for image, file in images:
        match = re.search('([fFbB])_([a-zA-Z0-9 ]+)(_.*)?', file)
        if match is None:
            continue
        cv2.imshow(file, image)
        cv2.waitKey()
        name = match.group(2)
        track_id, count = name_to_id.get(name, (None, 0))
        if track_id is None:
            track_id = next_id
            next_id -= 1
        track = tracked_objects.get(track_id, None)
        if track is None:
            track = PersonTrack(track_id, cameras)
            track.reid()  # just to be sure
            track.idenify(name)
            tracked_objects[track_id] = track

        if match.group(1).lower() == 'f':  # face
            track.add_face_sample(image, math.inf, count % cameras, True)
        elif match.group(1).lower() == 'b':  # body
            track.add_body_sample(image, math.inf, count % cameras, True)


# get trackers from rectangles and image
def build_trackers(rectangles, image):
    trackers = []
    for (x1, y1, x2, y2) in rectangles:
        tracker = dlib.correlation_tracker()
        rectangle = dlib.rectangle(x1, y1, x2, y2)
        tracker.start_track(image, rectangle)
        trackers.append(tracker)
    return trackers


def init():
    img_list_paths = list(map(lambda c: os.path.join(c, 'all_file.txt'), cams))
    print(img_list_paths)

    players = list(map(
        lambda l: PicturePlayback([os.path.join(os.path.dirname(l), line.rstrip('\n')) for line in open(l)], 30, True),
        img_list_paths))
    # players = [CameraPlayback()]
    # player = [VideoPlayback('video.avi')]
    # players = [YoutubePlayback('https://www.youtube.com/watch?v=N79f1znMWQ8')]

    # centroid tracker for each camera to keep track of IDs after new detection
    centroid_tracker = [CentroidTracker(0, 200) for _ in range(len(cams))]
    correlation_trackers = [[] for _ in range(len(cams))]

    for player in players:
        player.start()

    frame_index = 0
    while all([player.is_playing() for player in players]):
        frames = [player.get_frame() for player in players]
        if any(list(map(lambda f: f is None, frames))):
            print('Ended')
            break

        # current frame for each camera
        for camera_i, frame in enumerate(frames):
            # rectangles of detected people in current frame
            frame_copy = frame.copy()
            people = []
            should_detect = (frame_index % config.detect_frequency == 0)
            should_sample = (frame_index % config.detect_frequency == 0)
            should_reid = (frame_index % config.detect_frequency == 0)
            if should_detect:
                # detect rectangles
                people = api_functions.detect_people(frame)
                correlation_trackers[camera_i] = build_trackers(people, frame)
            else:
                # track rectangles
                for tracker in correlation_trackers[camera_i]:
                    tracker.update(frame)
                    pos = tracker.get_position()
                    people.append((int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())))
            # get rectangles with IDs assigned to them
            detected_objects = centroid_tracker[camera_i].update(people, should_detect)
            for (track_id, (x1, y1, x2, y2)) in detected_objects.items():
                # fix out of image
                x2, x1, y2, y1 = min(x2, frame.shape[1]), max(x1, 0), min(y2, frame.shape[1]), max(y1, 0)
                # too small, ignore
                if x2-x1 < 10 or y2 - y1 < 20:
                    continue
                # convert from internal track_id to actual person_id
                track_id = tracked_objects_reid.get(track_id, track_id)
                person_track = tracked_objects.get(track_id, None)

                cropped_body = frame[y1:y2, x1:x2]
                need_reid = should_reid
                if person_track is None:
                    if should_detect:  # so its not spammed when we dont keep track of all people
                        print('NEW PERSON DETECTED, CAMERA ' + str(camera_i))
                    person_track = PersonTrack(track_id, len(cams))
                    tracked_objects[track_id] = person_track
                    should_sample = True
                    need_reid = True  # check whether we have seen this person before
                else:
                    need_reid = need_reid and not person_track.was_reided()
                if should_sample:
                    person_track.add_body_sample(cropped_body, frame_index, camera_i)
                    face = api_functions.get_face(cropped_body)
                    if face is not None:
                        person_track.add_face_sample(face, frame_index, camera_i)
                if need_reid:
                    same_person_id = api_functions.compare_to_detected(person_track, tracked_objects)
                    if same_person_id is not None and same_person_id != track_id:
                        same_person_track = tracked_objects.get(same_person_id)
                        same_person_track.merge(person_track)
                        del tracked_objects[track_id]
                        tracked_objects_reid[track_id] = same_person_id
                        print(track_id, same_person_id)
                        track_id = same_person_id
                        person_track = same_person_track
                        person_track.reid()
                    elif same_person_id == track_id:  # this is an error and should never happened if everything is OK
                        print('ERROR comparing to same person ' + str(track_id))
                    # we do not keep track of this person, delete him
                    # TODO: creating the instance is quite unnecessary, but not 'that' costly...
                    if not config.keep_track_all and not person_track.is_known():
                        del tracked_objects[track_id]

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame_copy, person_track.get_name(), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Camera ' + str(camera_i), frame_copy)

        if cv2.waitKey(1)  == ord('q'):
            break

        frame_index += 1


    cv2.destroyAllWindows()


if __name__ == '__main__':
    if config.keep_track_targeted:
        build_known_people()
    init()