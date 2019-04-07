import help_functions
import numpy as np
import os

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform, RandomNormal

from keras import regularizers, backend

import config

import logging


# optimal init values http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
# maybe change later
def get_model(dimensions):
    network = Sequential()
    network.add(Conv2D(64, (10, 10), activation='relu', input_shape=dimensions,
                kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer='random_uniform',
                bias_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer='random_uniform',
                bias_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer='random_uniform',
                bias_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.0002)))
    network.add(Flatten())
    network.add(Dense(4096, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001),
                kernel_initializer='random_uniform', bias_initializer='random_uniform'))

    input1 = Input(dimensions)
    input2 = Input(dimensions)

    feature1 = network(input1)
    feature2 = network(input2)

    l1_layer = Lambda(lambda tensors: backend.abs(tensors[0] - tensors[1]))
    l1_distance = l1_layer([feature1, feature2])

    prediction = Dense(1, activation='sigmoid')(l1_distance)
    siamese_net = Model(inputs=[input1, input2], outputs=prediction)

    optimizer = Adam(0.001, decay=2.5e-4)
    siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return siamese_net


def get_image_pair_batch(people_count=10, folder=config.train_folder):
    while True:
        same, different = help_functions.get_image_pairs(folder, people_count)
        inputs, targets = help_functions.pairs_prepare(same, different)
        yield inputs, targets


def train():
    train_images = help_functions.load_all_images(config.train_folder, preprocess=help_functions.resize)

    # Hyper-parameters
    people_count = 10
    iterations = 60000
    checkpoint = 20
    save_checkpoint = 10000

    backend.clear_session()
    model = get_model((config.body_image_resize[1], config.body_image_resize[0], 3))
    f = open(os.path.join('model_history', 'perf.txt'), 'a')
    # print(model.summary())
    for i in range(1, iterations+1):
        same, different = help_functions.get_image_pairs(train_images, people_count)
        inputs, targets = help_functions.pairs_prepare(same, different)
        (loss, acc) = model.train_on_batch([inputs[0], inputs[1]], targets)

        if i % checkpoint == 0:
            print('Iteration: ' + str(i))
            print('Loss: ' + str(loss))
            print('Accuracy: ' + str(acc))
            f.write(str(i) + ' ' + str(loss) + ' ' + str(acc) + '\n')
        if i % save_checkpoint == 0:
            model.save_weights(os.path.join('model_history', str(i) + 'FB.h5'))
            logging.debug('model saved')
    f.close()


train()