import config
import numpy as np

import help_functions


def _add_sample(array, array_time, image, time, camera):
    start_index = config.sample_size * camera
    insert_index = int(np.argmin(array_time[start_index:start_index + config.sample_size]))
    array_time[start_index + insert_index] = time + 1
    array[start_index + insert_index] = image


class PersonTrack:
    def __init__(self, person_id, n_cameras, add_samples=True):
        self.history = [[] for _ in range(n_cameras)]
        self.id = person_id
        self.name = None

        # we identify by these
        self.body_samples = np.empty(
            (config.sample_size * n_cameras, config.body_image_resize[1], config.body_image_resize[0], 3))
        self.body_samples_times = np.random.rand(config.sample_size * n_cameras)

        self.face_samples = np.empty(
            (config.sample_size * n_cameras, config.face_image_resize[1], config.face_image_resize[0], 1))
        self.face_samples_times = np.random.rand(config.sample_size * n_cameras)

        # for building dataset or just displaying
        self.body_full_images = []
        self.face_full_images = []
        self.body_full_images_info = []
        self.face_full_images_info = []

        self.n_cameras = n_cameras
        self.reided = False

        self.add_samples = add_samples

    # add other track's info to this one
    def merge(self, same_person_track):
        if not self.is_known() and same_person_track.is_known():
            self.identify(same_person_track.get_name())

        # replace whatever was added later, however not the oldest photo may be replaces because times are not ordered
        replacing = self.body_samples_times < same_person_track.body_samples_times
        self.body_samples[replacing] = same_person_track.body_samples[replacing]
        self.body_samples_times[replacing] = same_person_track.body_samples_times[replacing]

        replacing = self.face_samples_times < same_person_track.face_samples_times
        self.face_samples[replacing] = same_person_track.face_samples[replacing]
        self.face_samples_times[replacing] = same_person_track.face_samples_times[replacing]

        # connect full image samples
        self.body_full_images = self.body_full_images + same_person_track.body_full_images
        self.face_full_images = self.face_full_images + same_person_track.face_full_images
        self.body_full_images_info = self.body_full_images_info + same_person_track.body_full_images_info
        self.face_full_images_info = self.face_full_images_info + same_person_track.face_full_images_info

    def is_known(self):
        return self.name is not None

    def identify(self, name):
        self.name = name

    # for storing history in camera
    def add_location(self, camera, location):
        self.history[camera].append(location)

    # if adding of samples is allowed, this image will be added even if it's older than all saved pictures
    def add_body_sample(self, image, time, camera, force_add=False):
        if config.keep_full_samples:
            self.body_full_images.append(image)
            self.body_full_images_info.append((time, camera))
        if self.add_samples or force_add:
            image = help_functions.prepare_body(image)
            _add_sample(self.body_samples, self.body_samples_times, image, time, camera)

    def add_face_sample(self, image, time, camera, force_add=False):
        if config.keep_full_samples:
            self.face_full_images.append(image)
            self.face_full_images_info.append((time, camera))
        if self.add_samples or force_add:
            image = help_functions.prepare_face(image)
            _add_sample(self.face_samples, self.face_samples_times, image, time, camera)

    def get_name(self):
        if not config.keep_track_all:
            return self.name or 'UNKNOWN'
        return self.name or ('ID: ' + str(self.id))

    def get_body_samples(self):
        return self.body_samples[self.body_samples_times >= 1]

    def get_face_samples(self):
        return self.face_samples[self.face_samples_times >= 1]

    def get_face_images(self):
        if config.keep_full_samples:
            return np.array(self.face_full_images)
        return self.get_face_samples()

    def get_body_images(self):
        if config.keep_full_samples:
            return np.array(self.body_full_images)
        return self.get_body_samples()

    def get_id(self):
        return self.id

    def was_reided(self):
        return self.reided

    def reid(self):
        self.reided = True
