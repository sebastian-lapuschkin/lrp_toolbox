#!/usr/bin/python
# -*- coding: utf-8 -*-

import modules
from modules import Sequential


class LRPModel(Sequential):
    '''
    Class to use LRP efficiently
    LRPModel loads a keras model to use LRP for the network
    '''

    def __init__(self, modules):
        super(LRPModel, self).__init__(modules)

    def evaluate(self, X, Y):
        '''
        Calculates the accuracy of the model from given dataset

        Parameters
        ----------
        X : numpy.ndarray
            a network input.
        Y : numpy.ndarray
            a ground truth dataset.

        Returns
        -------
        accuracy : float
            the accuracy of the calculation
        '''
        ypred = self.forward(X)
        num_true = (Y.argmax(axis=1) == ypred.argmax(axis=1)).sum()
        num_samples = Y.shape[0]
        accuracy = num_true / float(num_samples)
        return accuracy


def get_lrpmodule(layer):
    layer_name = layer.__class__.__name__
    modules = {
        "Conv2D": conv2d,
        "Dense": dense,
        "AveragePooling2D": averagepooling2D,
        "MaxPooling2D": maxpooling2D,
        "GlobalAveragePooling2D": globalaveragepooling2D,
        "Activation": activation,
        "Flatten": flatten,
        "InputLayer": nonlayer,
        "Dropout": nonlayer,
    }
    return modules[layer_name](layer)


def get_activation_lrpmodule(activation_layer):
    layer_name = activation_layer.__name__
    activation_modules = {
        "linear": None,
        "relu": modules.Rect(),
        "softmax": modules.SoftMax(),
    }
    return activation_modules[layer_name]


def conv2d(layer):
    h, w = layer.get_config()['kernel_size']
    d = layer.input_shape[-1]
    n = layer.output_shape[-1]
    s0, s1 = layer.get_config()['strides']
    W = layer.get_weights()[0]
    B = layer.get_weights()[1]
    module = modules.Convolution(filtersize=(h, w, d, n), stride=(s0, s1))
    module.W = W
    module.B = B
    activation_module = get_activation_lrpmodule(layer.activation)
    return module, activation_module


def dense(layer):
    m = layer.input_shape[-1]
    n = layer.output_shape[-1]
    module = modules.Linear(m, n)
    W, B = layer.get_weights()
    module.W = W
    module.B = B
    activation_module = get_activation_lrpmodule(layer.activation)
    return module, activation_module


def averagepooling2D(layer):
    h, w = layer.get_config()['pool_size']
    s0, s1 = layer.get_config()['strides']
    module = modules.AveragePool(pool=(h, w), stride=(s0, s1))
    return module, None


def maxpooling2D(layer):
    h, w = layer.get_config()['pool_size']
    s0, s1 = layer.get_config()['strides']
    module = modules.MaxPool(pool=(h, w), stride=(s0, s1))
    return module, None


def globalaveragepooling2D(layer):
    h, w = layer.input_shape[1:3]
    module = modules.AveragePool(pool=(h, w), stride=(1, 1))
    return module, modules.Flatten()


def activation(layer):
    module = get_activation_lrpmodule(layer.activation)
    return module, None


def flatten(layer):
    module = modules.Flatten()
    return module, None


def nonlayer(layer):
    return None, None


def read_kerasmodel(model):
    """
    keras modelを読み込んでLRPデモのモデルで返す
    """

    lrpmodules = []

    for layer in model.layers:
        module, activation_module = get_lrpmodule(layer)
        if module is not None:
            lrpmodules.append(module)
        if activation_module is not None:
            lrpmodules.append(activation_module)
    return LRPModel(lrpmodules)
