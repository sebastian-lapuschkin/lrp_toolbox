#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.layers import Conv2D, AvgPool2D, GlobalAveragePooling2D, MaxPooling2D, MaxPool2D, Activation, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.regularizers import l2


def lrp_demo_model(input_shape):
    """LeNet-5"""
    input_layer = Input(shape=input_shape, name='input1')

    layers = Conv2D(
        filters=10,
        kernel_size=5,
        padding='valid',
        activation='relu',
        name='conv1')(input_layer)
    layers = AvgPool2D(
        pool_size=2, strides=2, padding='valid', name='pool1')(layers)
    layers = Conv2D(
        filters=25,
        kernel_size=5,
        padding='valid',
        activation='relu',
        name='conv2')(layers)
    layers = AvgPool2D(
        pool_size=2, strides=2, padding='valid', name='pool2')(layers)
    layers = Conv2D(
        filters=100,
        kernel_size=4,
        padding='valid',
        activation='relu',
        name='conv3')(layers)
    layers = AvgPool2D(
        pool_size=2, strides=2, padding='valid', name='pool3')(layers)
    layers = Conv2D(
        filters=10,
        kernel_size=1,
        padding='valid',
        activation='relu',
        name='conv4')(layers)
    layers = Flatten(name='accuracy')(layers)
    layers = Activation('softmax', name='activation1')(layers)

    return Model(inputs=input_layer, outputs=layers)


def lrp_maxnet(input_shape):
    """LeNet-5 max pooling"""
    input_layer = Input(shape=input_shape, name='input1')

    layers = Conv2D(
        filters=10,
        kernel_size=5,
        padding='valid',
        activation='relu',
        name='conv1')(input_layer)
    layers = MaxPooling2D(
        pool_size=2, strides=2, padding='valid', name='pool1')(layers)
    layers = Conv2D(
        filters=25,
        kernel_size=5,
        padding='valid',
        activation='relu',
        name='conv2')(layers)
    layers = MaxPooling2D(
        pool_size=2, strides=2, padding='valid', name='pool2')(layers)
    layers = Conv2D(
        filters=100,
        kernel_size=4,
        padding='valid',
        activation='relu',
        name='conv3')(layers)
    layers = MaxPooling2D(
        pool_size=2, strides=2, padding='valid', name='pool3')(layers)
    layers = Conv2D(
        filters=10,
        kernel_size=1,
        padding='valid',
        activation='relu',
        name='conv4')(layers)
    layers = Flatten(name='accuracy')(layers)
    layers = Activation('softmax', name='activation1')(layers)

    return Model(inputs=input_layer, outputs=layers)
