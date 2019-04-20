import os

train_folder = os.path.join('DukeMTMC-reID', 'bounding_box_train')
train_output = os.path.join('DukeMTMC-reID', 'train_features.txt')

test_folder = os.path.join('DukeMTMC-reID', 'bounding_box_test')
test_output = os.path.join('DukeMTMC-reID', 'test_features.txt')

query_folder = os.path.join('DukeMTMC-reID', 'query')
query_output = os.path.join('DukeMTMC-reID', 'query_features.txt')

chokepoint = 'ChokePoint'
chokepoint_cropped_train = os.path.join(chokepoint, 'cropped_train')
chokepoint_cropped_test = os.path.join(chokepoint, 'cropped_test')
chokepoint_full = os.path.join(chokepoint, 'full')

att_faces = 'att_faces'

body_image_resize = (65, 155)
face_resize = (96, 96)

sample_size = 4

detect_frequency = 30

base_body_mode = os.path.join('model_history', 'base_body_weights.h5')
base_face_model = os.path.join('model_history', 'base_face_weights.h5')

improved_body_mode = os.path.join('model_history', 'improved_body_weights.h5')
improved_face_model = os.path.join('model_history', 'improved_face_weights.h5')

net_proto = os.path.join('mobilenet_ssd', 'MobileNetSSD_deploy.prototxt')
net_model = os.path.join('mobilenet_ssd', 'MobileNetSSD_deploy.caffemodel')

detect_body_confidence = 0.50

keep_track_all = True
keep_track_targeted = True
keep_track_targeted_files = 'known_people'





learning_start = True
learning_improving = False

avoid_false_positives = False

production = False
# user confirm before matching
confirm_match = True
# set to any number x to confirm x matches
# set to 0 to confirm  all matches that fit condition required_match_count
confirm_match_count = 3

# how many images have to be set as matched to be a match
required_match_count = 0.66

keep_full_samples = True
