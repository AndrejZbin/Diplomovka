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

body_image_resize = (65, 155)
face_resize = (96, 96)

sample_size = 16
