import config
import random
import numpy as np
import cv2

import retinex
import api_functions


class PersonTrack:
    def __init__(self, person_id, n_cameras, add_samples=True):
        self.history = [[] for _ in range(n_cameras)]
        self.id = person_id
        self.name = None

        self.body_samples = np.empty(
            (config.sample_size * n_cameras, config.body_image_resize[1], config.body_image_resize[0], 3))
        self.body_samples_times = np.zeros((config.sample_size * n_cameras))

        self.face_samples = np.empty(
            (config.sample_size * n_cameras, config.face_resize[1], config.face_resize[0], 1))
        self.face_samples_times = np.zeros((config.sample_size * n_cameras))

        self.n_cameras = n_cameras
        self.reided = False

        self.add_samples = add_samples

    def _add_sample(self, array, array_time, image, time, camera):
        start_index = config.sample_size*camera
        insert_index = int(np.argmin(array_time[start_index:start_index + config.sample_size]))
        array_time[start_index + insert_index] = time + 1
        array[start_index + insert_index] = image

    def merge(self, same_person):
        if not self.is_known() and same_person.is_known():
            self.idenify(same_person.get_name())

        # replace whatever was added later, however not the oldest photo may be replaces because times are not ordered
        replacing = self.body_samples_times < same_person.body_samples_times
        self.body_samples[replacing] = same_person.body_samples[replacing]

        replacing = self.face_samples_times < same_person.face_samples_times
        self.face_samples[replacing] = same_person.face_samples[replacing]

    def is_known(self):
        return self.name is not None

    def idenify(self, name):
        self.name = name

    def add_location(self, camera, location):
        self.history[camera].append(location)

    # if adding of samples is allowed, this image will be added even if it's older than all saved pictures
    def add_body_sample(self, image, time, camera, force_add=False):
        if self.add_samples or force_add:
            image = api_functions.prepare_body(image)
            self._add_sample(self.body_samples, self.body_samples_times, image, time, camera)

    def add_face_sample(self, image, time, camera, force_add=False):
        if self.add_samples or force_add:
            image = api_functions.prepare_face(image)
            self._add_sample(self.face_samples, self.face_samples_times, image, time, camera)

    def get_name(self):
        if not config.keep_track_all:
            return self.name or 'UNKNOWN'
        return self.name or ('UNKNOWN ID: ' + str(self.id))

    def get_body_samples(self):
        return self.body_samples[self.body_samples_times.astype(bool)]

    def get_face_samples(self):
        return self.face_samples[self.face_samples_times.astype(bool)]

    def get_id(self):
        return self.id

    def was_reided(self):
        return self.reided

    def reid(self):
        self.reided = True







