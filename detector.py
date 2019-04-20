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

import siamese_network

import numpy as np

import logging

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
    images = help_functions.load_all_images(config.keep_track_targeted_files, '', help_functions.resize_face)
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
        track = None
        # first time we see picture of this person
        if track_id is None:
            track_id = next_id
            next_id -= 1
            track = PersonTrack(track_id, n_cameras)
            track.reid()  # not necessary but good to have
            track.idenify(name)
            tracked_objects[track_id] = track
            name_to_id[name] = (track_id, count +1)
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
    for cams, group_name in cams_groups:
        # delete everything we know
        tracked_objects.clear()
        tracked_objects_reid.clear()
        if config.keep_track_targeted:
            build_known_people()

        # TODO: don't use all_file.txt file, instead load images using functions made for it
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

        # start playback
        for player in players:
            player.start()

        frame_index = 0
        while all([player.is_playing() for player in players]):
            frames = [player.get_frame() for player in players]
            # make them all to have the same length
            if any(list(map(lambda f: f is None, frames))):
                print('Ended')
                break

            # current frame for each camera
            for camera_i, frame in enumerate(frames):
                # rectangles of detected people in current frame
                frame_copy = frame.copy()
                bodies = []
                # should we try detecting again?
                should_detect = (frame_index % config.detect_frequency == 0)
                # should face and body samples be saved from the current frame?
                should_sample = (frame_index % config.detect_frequency == 0)
                # should we try to find match for newly detected people?
                should_reid = (frame_index % config.detect_frequency == 0)
                if should_detect:
                    # detect rectangles
                    bodies = api_functions.detect_people(frame)
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
                    need_reid = should_reid
                    if person_track is None:
                        if should_detect:  # so its not spammed when we dont keep track of all people
                            logging.info('PLAYBACK: new person {} detected in camera {}'.format(track_id, camera_i))
                        person_track = PersonTrack(track_id, n_cameras)
                        tracked_objects[track_id] = person_track
                        # sample for re-ID
                        should_sample = True
                        # try to find whether we have seen this person before
                        need_reid = True
                    else:
                        # don't re-ID people who were re-IDed before, so there are no cycles in detection
                        # all detections of same person will be matched eventually
                        # TODO: re-IDed person should be re-IDed again, because A1==A2 =never match= B1==B2
                        need_reid = need_reid and not person_track.was_reided()
                    if should_sample:
                        person_track.add_body_sample(cropped_body, frame_index, camera_i)
                        # try to find face of this person
                        face = api_functions.get_face(cropped_body)
                        if face is not None:
                            person_track.add_face_sample(face, frame_index, camera_i)
                    if need_reid:
                        same_person_id = api_functions.compare_to_detected(person_track, tracked_objects)
                        if same_person_id is not None and same_person_id != track_id:
                            # get track of person we matched
                            same_person_track = tracked_objects.get(same_person_id)
                            # merge information
                            same_person_track.merge(person_track)
                            # we only need one track, the one that doesn't have less information
                            del tracked_objects[track_id]
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
            build_new_dataset(group_name)

    logging.info('PLAYBACK: Playback finished')


# build a new dataset from information we gathered from this playback
def build_new_dataset(group_name):
    # we save images here
    path = os.path.join(config.improve_folder, group_name)
    # create necessary folders
    if not os.path.exists(os.path.join(path, 'faces')):
        os.makedirs(os.path.join(path, 'faces'))
    if not os.path.exists(os.path.join(path, 'bodies')):
        os.makedirs(os.path.join(path, 'bodies'))
    # iterate over tracked people
    for track_id, track in tracked_objects.items():
        # if people were unmatched, they could cause false negatives, we don't want feed our model with that
        if not track.was_reided():
            continue
        faces = track.face_full_images
        faces_info = track.face_full_images_info

        bodies = track.body_full_images
        bodies_info = track.body_full_images_info
        for i in range(len(faces)):
            face = faces[i]
            frame, camera = faces_info[i]
            # make filename matching other dataset's filenames
            filename = str(track_id) + '_c' + str(camera) + '_f' + str(frame) + '.jpg'
            # finally save image
            cv2.imwrite(os.path.join(path, 'faces', filename), face)
        for i in range(len(bodies)):
            body = bodies[i]
            frame, camera = bodies_info[i]
            # make filename matching other dataset's filenames
            filename = str(track_id) + '_c' + str(camera) + '_f' + str(frame) + '.jpg'
            # finally save image
            cv2.imwrite(os.path.join(path, 'bodies', filename), body)


# fix dataset by matching different IDs, check manually, also delete manually bad files
def fix_built_dataset():
    while True:
        r = input('Fix from to: ')
        r = r.split(' ')
        if len(r) != 2:
            break
        try:
            f, t = int(r[0]), int(r[1])
        except ValueError:
            break
        for path, subdirs, files in os.walk(config.improve_folder):
            for file in files:
                person_id, camera, frame = help_functions.info_from_image_name(file)
                if person_id == f:
                    filename = str(t) + '_c' + str(camera) + '_f' + str(frame) + '.jpg'
                    os.rename(os.path.join(path, file), os.path.join(path, filename))


# TODO: DRY, code is repeated for faces and bodies
# improve model for recognizing faces by learning from captured data
# potential problem A1 matched with A2, B1 matched with B2, but A1,B1,A2,B2 is one person
def improve_faces():
    # same person in different folders might have different ID, causing false negatives if we connected them
    train_images = []
    # load images from each folder
    for group_name in config.improve_camera_groups:
        images = help_functions.load_all_images(os.path.join(config.improve_folder, group_name, 'faces'),
                                                file_type='.jpg', preprocess=help_functions.resize_face)
        train_images.append(images)

    # hyper-parameters
    people_count = 8
    iterations = 1000
    checkpoint = 20
    save_checkpoint = 5000

    model = siamese_network.get_face_model((config.face_resize[1], config.face_resize[0], 1))

    # are we improving base model of already improved model?
    # load weight for model we are improving
    if config.learning_start:
        model.load_weights(filepath=config.base_face_model)
    elif config.learning_improving:
        model.load_weights(filepath=config.improved_face_model)

    f = open(os.path.join('model_history', 'face_improve_perf.txt'), 'a')
    logging.info('IMPROVING: Starting to improve model for faces')
    for i in range(1, iterations+1):
        inputs, targets = help_functions.get_image_pairs(train_images[np.random.randint(0, len(train_images))], people_count)
        (loss, acc) = model.train_on_batch(inputs, targets)
        if i % checkpoint == 0:
            logging.info('Iteration: {}'.format(i))
            logging.info('Loss: {}'.format(loss))
            logging.info('Accuracy: {}'.format(acc))
            f.write(str(i) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        if i % save_checkpoint == 0:
            model.save_weights(os.path.join('model_history', str(i) + 'FI.h5'))
            f.flush()
    model.save_weights(config.improved_face_model)
    f.close()


# improve model for recognizing bodies by learning from captured data
def improve_bodies():
    # same person in different folders might have different ID, causing false negatives if we connected them
    train_images = []
    # load images from each folder
    for group_name in config.improve_camera_groups:
        images = help_functions.load_all_images(os.path.join(config.improve_folder, group_name, 'bodies'),
                                                file_type='.jpg', preprocess=help_functions.resize_body)
        train_images.append(images)

    # hyper-parameters
    people_count = 8
    iterations = 1000
    checkpoint = 20
    save_checkpoint = 5000

    model = siamese_network.get_body_model((config.body_image_resize[1], config.body_image_resize[0], 3))

    # are we improving base model of already improved model?
    # load weight for model we are improving
    if config.learning_start:
        model.load_weights(filepath=config.base_body_model)
    elif config.learning_improving:
        model.load_weights(filepath=config.improved_body_model)

    f = open(os.path.join('model_history', 'face_improve_perf.txt'), 'a')
    logging.info('IMPROVING: Starting to improve model for bodies')
    for i in range(1, iterations+1):
        inputs, targets = help_functions.get_image_pairs(train_images[np.random.randint(0, len(train_images))], people_count)
        (loss, acc) = model.train_on_batch(inputs, targets)
        if i % checkpoint == 0:
            logging.info('Iteration: {}'.format(i))
            logging.info('Loss: {}'.format(loss))
            logging.info('Accuracy: {}'.format(acc))
            f.write(str(i) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        if i % save_checkpoint == 0:
            model.save_weights(os.path.join('model_history', str(i) + 'FBI.h5'))
            f.flush()
    model.save_weights(config.improved_body_model)
    f.close()


if __name__ == '__main__':
    playback()

    # finally we begin learning on newly collected information
    if config.learning_improving or config.learning_start:
        fix_built_dataset()
        improve_faces()
        improve_bodies()
