#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import numpy as np
from nn_utils import shuffle

from kerasmodels import lrp_demo_model, lrp_maxnet


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., np.newaxis]
    x_train = x_train / 127.5 - 1.
    x_train = np.pad(
        x_train,
        ((0, 0), (2, 2), (2, 2), (0, 0)),
        'constant',
        constant_values=(-1., ),
    )

    x_test = x_test[..., np.newaxis]
    x_test = x_test / 127.5 - 1.
    x_test = np.pad(
        x_test,
        ((0, 0), (2, 2), (2, 2), (0, 0)),
        'constant',
        constant_values=(-1., ),
    )

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    x_train, y_train = shuffle(x_train, y_train)

    tensorlog_dir = './logs/tensorboard'
    if tf.gfile.Exists(tensorlog_dir):
        tf.gfile.DeleteRecursively(tensorlog_dir)

    model = lrp_demo_model(input_shape=x_train[0].shape)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'],
    )
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        verbose=1,
        epochs=100,
        validation_data=(x_test, y_test),
        shuffle=True,
    )
    acc = model.evaluate(x_test, y_test)[1]

    trained_models_dir = './models/'
    model_filename = os.path.join(trained_models_dir, 'lrp_demo_mnist.h5')
    model.save(model_filename)


if __name__ == "__main__":
    main()
