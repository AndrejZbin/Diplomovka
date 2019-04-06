import os

train_folder = os.path.join('DukeMTMC-reID', 'bounding_box_train')
train_output = os.path.join('DukeMTMC-reID', 'train_features.txt')

test_folder = os.path.join('DukeMTMC-reID', 'bounding_box_test')
test_output = os.path.join('DukeMTMC-reID', 'test_features.txt')

query_folder = os.path.join('DukeMTMC-reID', 'query')
query_output = os.path.join('DukeMTMC-reID', 'query_features.txt')

body_image_resize = (65, 155)
