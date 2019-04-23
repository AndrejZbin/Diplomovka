import re
import math
import dlib
import logging

import detect
import help_functions
import recognize
import config
import improve

from centroid_tracker import CentroidTracker
from person_track import PersonTrack

from players import *

# saved information about detected people, key=ID, value=person_track instance
tracked_objects = {}
# id from tracker -> id of person
tracked_objects_reid = {}

# how many cameras are there
n_cameras = config.n_cameras

# camera groups, put all cameras to one group if not learning
# each group should have n_cameras cameras
cams_groups = config.cams_groups


# insert known people into tracked_objects dict so we can recognize them
def build_known_people():
    # not necessary to know people
    if config.learning_start or config.learning_improving:
        return
    images = help_functions.load_all_images(config.keep_track_targeted_files, '', help_functions.identity)
    # one id for one name, there may be multiple files with same them
    name_to_id = {}

    # make IDs negative so we don't have to worry about collisions with centroid tracker
    # TODO: hack-y way, instead make centroid tracker next_id larger so there are no collision, but this works OK
    next_id = -1
    for image, file in images:
        # name format F_person name_.... for faces, B_person name for bodies
        match = re.search('([fFbB])_([a-zA-Z0-9 ]+)(_.*)?', file)
        # ignore wrong files
        if match is None:
            logging.warning('KNOWN PEOPLE: File {} has wrong format'.format(file))
            continue
        name = match.group(2)
        # try to get track if we have seen this person before and how many time we have seen him
        track_id, count = name_to_id.get(name, (None, 0))
        # first time we see picture of this person
        if track_id is None:
            track_id = next_id
            next_id -= 1
            track = PersonTrack(track_id, n_cameras)
            track.reid()  # not necessary but good to have
            track.identify(name)
            tracked_objects[track_id] = track
            name_to_id[name] = (track_id, count + 1)
        else:
            track = tracked_objects.get(track_id, None)

        # distribute pictures to different cameras, but it doesn't really matter that much
        if match.group(1).lower() == 'f':  # face
            track.add_face_sample(image, math.inf, count % n_cameras, True)
        elif match.group(1).lower() == 'b':  # body
            track.add_body_sample(image, math.inf, count % n_cameras, True)


# get trackers from rectangles and image
# rectangles should contain full coordinates, not with and height
def build_trackers(rectangles, image):
    trackers = []
    for (x1, y1, x2, y2) in rectangles:
        tracker = dlib.correlation_tracker()
        rectangle = dlib.rectangle(x1, y1, x2, y2)
        tracker.start_track(image, rectangle)
        trackers.append(tracker)
    return trackers


# start playing video
def playback():
    # handle each group separately, good for making DB
    for cams, group_name, fps in cams_groups:
        # delete everything we know
        tracked_objects.clear()
        tracked_objects_reid.clear()
        if config.keep_track_targeted:
            build_known_people()

        players = [PicturePlayback(camera, fps, not config.playback_realtime) for camera in cams]
        # players = [CameraPlayback()]
        # player = [VideoPlayback('video.avi')]
        # players = [YoutubePlayback('https://www.youtube.com/watch?v=N79f1znMWQ8')]

        # centroid tracker for each camera to keep track of IDs after new detection
        centroid_tracker = [CentroidTracker(0, config.centroid_max_distance) for _ in range(len(cams))]
        correlation_trackers = [[] for _ in range(len(cams))]

        # start playback
        for player in players:
            player.start()

        frame_index = 0
        while any([player.is_playing() for player in players]):
            frames = [player.get_frame() for player in players]
            # make them all to have the same length
            # current frame for each camera
            for camera_i, frame in enumerate(frames):
                if frame is None:
                    continue
                # rectangles of detected people in current frame
                frame_copy = frame.copy()
                bodies = []
                # should we try detecting again?
                should_detect = (frame_index % config.detect_frequency == 0)
                if should_detect:
                    # detect rectangles
                    bodies = detect.detect_people(frame)
                    correlation_trackers[camera_i] = build_trackers(bodies, frame)
                else:
                    # track rectangles
                    for tracker in correlation_trackers[camera_i]:
                        tracker.update(frame)
                        pos = tracker.get_position()
                        bodies.append((int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())))
                # get rectangles with IDs assigned to them
                detected_objects = centroid_tracker[camera_i].update(bodies, should_detect)
                for (track_id, (x1, y1, x2, y2)) in detected_objects.items():
                    # should face and body samples be saved from the current frame?
                    should_sample = (frame_index % config.detect_frequency == 0)
                    # should we try to find match for newly detected people?
                    should_reid = (frame_index % config.detect_frequency == 0)
                    # TODO: clear these checks
                    # fix out of image
                    x2, x1, y2, y1 = min(x2, frame.shape[1]), max(x1, 0), min(y2, frame.shape[1]), max(y1, 0)
                    # too small, ignore
                    if x2-x1 < 10 or y2 - y1 < 20:
                        continue
                    # convert from internal track_id to actual person_id
                    track_id = tracked_objects_reid.get(track_id, track_id)
                    person_track = tracked_objects.get(track_id, None)

                    cropped_body = frame[y1:y2, x1:x2]
                    if should_detect:
                        if person_track is None:
                            logging.info('PLAYBACK: new person {} detected in camera {}'.format(track_id, camera_i))
                            person_track = PersonTrack(track_id, n_cameras)
                            tracked_objects[track_id] = person_track
                            # sample for re-ID
                            should_sample = True
                            # try to find whether we have seen this person before
                            should_reid = True
                        else:
                            # compare to self just in case it's actually a new person
                            test_id = centroid_tracker[camera_i].next_id
                            test_track = PersonTrack(test_id, n_cameras)
                            test_track.add_body_sample(cropped_body, frame_index, camera_i)
                            face = detect.get_face(cropped_body)
                            if face is not None:
                                test_track.add_face_sample(face, frame_index, camera_i)
                            self_compare = recognize.compare_to_detected(test_track, {track_id: person_track})
                            # same centroid but persons don't match
                            if self_compare is None:
                                # re-id this centroid because i'ts not the same person
                                centroid_tracker[camera_i].reid(track_id, test_id)
                                CentroidTracker.next_id += 1

                                tracked_objects[test_id] = test_track
                                track_id = test_id
                                person_track = test_track

                                logging.info('PLAYBACK: new person {} detected in camera {}'.format(track_id, camera_i))

                                should_sample = False
                                should_reid = True
                            # TODO: re-IDed person should be re-IDed again, because A1==A2 =never match= B1==B2
                            # don't re-ID people who were re-IDed before, so there are no cycles in detection
                            # all detections of same person will be matched eventually
                            else:
                                should_reid = should_reid and not person_track.was_reided()

                    if should_sample:
                        person_track.add_body_sample(cropped_body, frame_index, camera_i)
                        # try to find face of this person
                        face = detect.get_face(cropped_body)
                        if face is not None:
                            person_track.add_face_sample(face, frame_index, camera_i)
                    if should_reid:
                        same_person_id = recognize.compare_to_detected(person_track, tracked_objects)
                        if same_person_id is not None and same_person_id != track_id:
                            # get track of person we matched
                            same_person_track = tracked_objects.get(same_person_id)
                            # merge information
                            same_person_track.merge(person_track)
                            # we only need one track, the one that doesn't have less information
                            tracked_objects.pop(track_id)
                            # re-ID from trackers ID to person ID
                            tracked_objects_reid[track_id] = same_person_id
                            # update values
                            track_id = same_person_id
                            person_track = same_person_track
                            person_track.reid()
                        elif same_person_id == track_id:  # this is an error and should never happened :)
                            logging.error(
                                'PLAYBACK: comparing and matching same person {}, something is wrong'.format(track_id))
                        # we do not keep track of this person, delete him
                        # TODO: creating the instance is quite unnecessary, but not 'that' costly...
                        if not config.keep_track_all and not person_track.is_known():
                            del tracked_objects[track_id]

                    # display information on screen
                    # TODO: maybe add face? but tracking it is unnecessary and we detect in irregularly, so probably not
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frame_copy, person_track.get_name(), (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('Camera {}'.format(camera_i), frame_copy)

            # TODO: better quitting but it's OK for now
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

        cv2.destroyAllWindows()

        # build database from collected information
        if config.build_dataset:
            improve.build_new_dataset(group_name, tracked_objects)

    logging.info('PLAYBACK: Playback finished')


if __name__ == '__main__':
    playback()
