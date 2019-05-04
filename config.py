import os
import logging

# some parameters are important for good accuracy, depending on camera position and other factors

# don't learn anything
production = True
# improve base model
learning_start = False
# improve improved model
learning_improving = False

build_dataset = True

playback_realtime = False

# should person be compares to himself every detection?
reid_same = False

# user confirm before matching
confirm_match = True
# set to any number x to confirm x matches
# set to 0 to confirm  all matches that fit condition required_match_count
confirm_match_count = 0

n_cameras = 3
cams_groups = [(['P2E_S4_C1.1', 'P2E_S4_C2.1', 'P2E_S3_C3.1'], 'group1', 30)]


# all folders
train_body_folder = os.path.join('DukeMTMC-reID', 'bounding_box_train')
train_body_features = os.path.join('DukeMTMC-reID', 'train_features.txt')

test_body_folder = os.path.join('DukeMTMC-reID', 'bounding_box_test')
test_body_features = os.path.join('DukeMTMC-reID', 'test_features.txt')

query_body_folder = os.path.join('DukeMTMC-reID', 'query')
query_body_features = os.path.join('DukeMTMC-reID', 'query_features.txt')

base_body_model = os.path.join('model_history', 'base_body_weights.h5')
base_face_model = os.path.join('model_history', 'base_face_weights.h5')

improved_body_model = os.path.join('model_history', 'improved_body_weights.h5')
improved_face_model = os.path.join('model_history', 'improved_face_weights.h5')

mobilenet_proto = os.path.join('mobilenet_ssd', 'MobileNetSSD_deploy.prototxt')
mobilenet_model = os.path.join('mobilenet_ssd', 'MobileNetSSD_deploy.caffemodel')

chokepoint = 'ChokePoint'
chokepoint_cropped_train = os.path.join(chokepoint, 'cropped_train')
chokepoint_cropped_test = os.path.join(chokepoint, 'cropped_test')
chokepoint_full = os.path.join(chokepoint, 'full')

att_faces = 'att_faces'

# for improving on new data
improve_folder = 'improve_dataset'
improve_camera_groups = ['group1']

# size for images fed to model
body_image_resize = (65, 155)
face_image_resize = (96, 96)

# how many samples for each camera is saved in track
sample_size = 4

# how often we detect again (30 means 1 in 30 displayed frames)
detect_frequency = 30

detect_body_confidence = 0.50

# assign ID to all people we detect and keep information about them to next detect?
keep_track_all = True
# are there any people we are looking for?
keep_track_targeted = True
# storage for images of people we are looking for
keep_track_targeted_files = 'known_people'

# how much can a person move between frame, set to 0 for comparing each frame
centroid_max_distance = 200

avoid_false_positives = False

# how many images have to be set as matched to be a match
required_match_count = 0.66

keep_full_samples = True

body_compare_trust = 1.0
face_compare_trust = 1.0

# ###############################################
#                                               #
# ###############################################

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-1s: %(message)s",
    datefmt="%H:%M:%S")

# 0 = NN, 1 = HOG, 2 = HAAR
body_detect_method = False

if production:
    learning_start = False
    learning_improving = False

if not confirm_match:
    keep_track_all = False
