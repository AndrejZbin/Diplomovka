import requests
import zipfile
import tarfile
import io
import os
import logging


def duke():
    url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    logging.debug('DukeMTMC ready')


def chokepoint():
    chokepoint_files = [
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
        'P2L_S5.tar.xz']

    if not os.path.exists('chokepoint'):
        os.mkdir('chokepoint')

    chokepoint_base_url = 'https://zenodo.org/record/815657/files/'

    for chokepoint_file in chokepoint_files:
        full_url = chokepoint_base_url + chokepoint_file
        r = requests.get(full_url)
        t = tarfile.open(fileobj=io.BytesIO(r.content), mode='r:xz')
        t.extractall(path='chokepoint')
        logging.debug(chokepoint_file + 'downloaded')

    logging.debug('ChokePoint ready')

chokepoint()