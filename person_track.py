import config
import random


def _add_sample(array, image):
    n_samples = len(array)
    if n_samples < config.sample_size:
        array.append(image)
    else:
        array[random.randint(0, n_samples - 1)] = image


class PersonTrack:
    def __init__(self, person_id, n_cameras):
        self.history = [[] for _ in range(n_cameras)]
        self.id = person_id
        self.name = None
        self.body_samples = [[] for _ in range(n_cameras)]
        self.face_samples = [[] for _ in range(n_cameras)]

    def is_known(self):
        return self.name is not None

    def idenify(self, name):
        self.name = name

    def add_location(self, camera, location):
        self.history[camera].append(location)

    def add_body_sample(self, image, camera):
        _add_sample(self.body_samples[camera], image)

    def add_face_sample(self, image, camera):
        _add_sample(self.face_samples[camera], image)

    def get_name(self):
        return self.name or ('UNKNOWN ID: ' + str(self.id))





