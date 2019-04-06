import help_functions
import numpy as np

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


# optimal init values http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
def random_weights():
    return RandomNormal(mean=0, stddev=0.01, seed=None)


def random_bias():
    return RandomNormal(mean=0.5, stddev=0.01, seed=None)


def get_model(dimensions):
    network = Sequential()
    network.add(Conv2D(64, (10, 10), activation='relu', input_shape=dimensions,
                kernel_initializer=random_weights(), kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=random_weights(),
                bias_initializer=random_bias(), kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=random_weights(),
                bias_initializer=random_bias(), kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=random_weights(),
                bias_initializer=random_bias(), kernel_regularizer=regularizers.l2(0.0002)))
    network.add(Flatten())
    network.add(Dense(4096, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001),
                kernel_initializer=random_weights(), bias_initializer=random_bias()))

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
    # RGB
    model = get_model((config.body_image_resize[1], config.body_image_resize[0], 3))
    # print(model.summary())

    # Hyper-parameters
    people_count = 10
    iterations = 200

    for i in range(iterations):
        same, different = help_functions.get_image_pairs(config.train_folder, people_count)
        inputs, targets = help_functions.pairs_prepare(same, different)
        (loss, acc) = model.train_on_batch([inputs[0], inputs[1]], targets)
        print('Loss: ' + str(loss))
        print('Accuracy: ' + str(acc))



train()



