import requests
import zipfile
import tarfile
import io
import os
import logging
import re
import random

import config


def duke():
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    logging.debug('DukeMTMC ready')


def att():
    url = 'http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(config.att_faces)
    logging.debug('ATT ready')

    for path, subdirs, files in os.walk('att_faces'):
        for file in files:
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path) and file.endswith('.pgm'):
                match = re.search('\\W+s([0-9]+)\\W+([0-9]+).pgm$', full_path)
                person, camera = match.group(1), match.group(2)
                filename = person + '_c' + camera + '_f1.pgm'
                os.rename(full_path, os.path.join(config.att_faces, filename))


# takes a lot of time to download and extract :(
def chokepoint():
    chokepoint_files = [
        # download these for full frames to play as video
        'P1E_S1.tar.xz',
        'P1E_S2.tar.xz',
        'P1E_S3.tar.xz',
        'P1E_S4.tar.xz',
        'P1L_S1.tar.xz',
        'P1L_S2.tar.xz',
        'P1L_S3.tar.xz',
        'P1L_S4.tar.xz',
        'P2E_S1.tar.xz',
        'P2E_S2.tar.xz',
        'P2E_S3.tar.xz',
        'P2E_S4.tar.xz',
        'P2E_S5.tar.xz',
        'P2L_S1.tar.xz',
        'P2L_S2.tar.xz',
        'P2L_S3.tar.xz',
        'P2L_S4.tar.xz',
        'P2L_S5.tar.xz',
        # cropped faces for training
        'P1E.tar.xz',
        'P1L.tar.xz',
        'P2E.tar.xz',
        'P2L.tar.xz',
    ]

    tmp1 = os.path.join(config.chokepoint, 'tmp1')
    tmp2 = os.path.join(config.chokepoint, 'tmp2')

    if not os.path.exists(config.chokepoint):
        os.mkdir(config.chokepoint)
    if not os.path.exists(config.chokepoint_cropped_train):
        os.mkdir(config.chokepoint_cropped_train)
    if not os.path.exists(config.chokepoint_cropped_test):
        os.mkdir(config.chokepoint_cropped_test)
    if not os.path.exists(config.chokepoint_full):
        os.mkdir(config.chokepoint_full)
    if not os.path.exists(tmp1):
        os.mkdir(tmp1)
    if not os.path.exists(tmp1):
        os.mkdir(tmp2)

    chokepoint_base_url = 'https://zenodo.org/record/815657/files/'

    for chokepoint_file in chokepoint_files:
        full_url = chokepoint_base_url + chokepoint_file
        r = requests.get(full_url)
        t = tarfile.open(fileobj=io.BytesIO(r.content), mode='r:xz')
        # cropped faces
        if len(chokepoint_file) == 10:
            t.extractall(path=tmp1)
        else:
            t.extractall(path=config.chokepoint)
        logging.debug(chokepoint_file + 'downloaded')

    for file in os.listdir(config.chokepoint):
        path = os.path.join(config.chokepoint, file)
        if os.path.isfile(path) and file.endswith('.tar.xz'):
            t = tarfile.open(path, mode='r:xz')
            t.extractall(path=tmp2)

    # moves files and renames them to same pattern as duke
    move_prepare_files(tmp1, config.chokepoint_cropped_train, config.chokepoint_cropped_test, '.pgm', decide_train)

    logging.debug('ChokePoint ready')


helper = {
    'P1E': 0,
    'P1L': 1,
    'P2E': 2,
    'P2L': 3,
    'num_cams': 3,
    'max_frames': 1000000
}


def decide_train(person, place, _, camera):
    if person in [4, 5, 7]:
        return False
    if place == 'P1E':
        return True
    if place == 'P1L':
        return False
    if place == 'P2E':
        return camera in [1, 2]
    if place == 'P2L':
        return camera == 3
    return random.randint(0, 1) == 0


def move_prepare_files(from_path, to_path_train, to_path_test, file_type, decision_f):
    for path, subdirs, files in os.walk(from_path):
        for file in files:
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path) and file.endswith(file_type):
                m = re.search(
                    '(P[0-9][LE])_S([0-9]+)_C([0-9]+)\\W+([0-9]+)\\W+([0-9]+)' + file_type + '$', full_path)
                place, seq, cam, person, frame = m.group(1), m.group(2),  m.group(3), m.group(4), m.group(5)
                filename = person + '_c' + str(int(cam) + helper[place]*helper['num_cams']) + \
                                    '_f' + str(int(frame) + (int(seq) - 1) * helper['max_frames']) + \
                                    file_type
                if decision_f(person, place, seq, cam):
                    os.rename(full_path, os.path.join(to_path_train, filename))
                else:
                    os.rename(full_path, os.path.join(to_path_test, filename))


if __name__ == '__main__':
    chokepoint()
    duke()
    att()
