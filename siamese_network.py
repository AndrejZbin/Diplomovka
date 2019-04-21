from keras import regularizers, backend
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Input, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense

import config


def get_face_model(dimensions=(config.face_image_resize[1], config.face_image_resize[0], 1)):
    network = Sequential()
    network.add(Conv2D(64, (10, 10), activation='relu', input_shape=dimensions,
                kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Dropout(.15))
    network.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer='random_uniform',
                bias_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Dropout(.15))
    network.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer='random_uniform',
                bias_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.0002)))
    network.add(MaxPooling2D())
    network.add(Dropout(.15))
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


def get_body_model(dimensions=(config.body_image_resize[1], config.body_image_resize[0], 3)):
    return get_face_model(dimensions)
