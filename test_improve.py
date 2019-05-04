import os

import test
import config

if __name__ == '__main__':
    print('NOT IMPROVED')
    test.test_body(config.base_body_model, os.path.join(config.improve_folder, 'group2', 'bodies'))
    test.test_face(config.base_face_model, os.path.join(config.improve_folder, 'group2', 'faces'), '.jpg')

    print('IMPROVED')
    test.test_body(config.improved_body_model, os.path.join(config.improve_folder, 'group2', 'bodies'))
    test.test_face(config.improved_face_model, os.path.join(config.improve_folder, 'group2', 'faces'), '.jpg')
