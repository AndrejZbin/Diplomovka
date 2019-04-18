import config
import random
import numpy as np
import cv2

import retinex
import api_functions


class PersonTrack:
    def __init__(self, person_id, n_cameras):
        self.history = [[] for _ in range(n_cameras)]
        self.id = person_id
        self.name = None
        self.body_samples = []
        self.face_samples = []
        self.n_cameras = n_cameras
        self.reided = False

    def _add_sample(self, array, image, camera):
        n_samples = len(array)
        if n_samples < config.sample_size * self.n_cameras:
            array.append(image)
        else:
            array[random.randint(config.sample_size*camera, config.sample_size*camera + config.sample_size - 1)] = image

    def is_known(self):
        return self.name is not None

    def idenify(self, name):
        self.name = name

    def add_location(self, camera, location):
        self.history[camera].append(location)

    def add_body_sample(self, image, camera):
        image = api_functions.prepare_body(image)
        self._add_sample(self.body_samples, image, camera)

    def add_face_sample(self, image, camera):
        image = api_functions.prepare_face(image)
        self._add_sample(self.face_samples, image, camera)

    def get_name(self):
        return self.name or ('UNKNOWN ID: ' + str(self.id))

    def get_body_samples(self):
        return np.array(self.body_samples)

    def get_face_samples(self):
        return np.array(self.face_samples)

    def get_id(self):
        return self.id

    def was_reided(self):
        return self.reided

    def reid(self):
        self.reided = True







