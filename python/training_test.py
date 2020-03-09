'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 30.09.2015
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import modules
import model_io
import data_io

import importlib.util as imp
import numpy
import numpy as np
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
na = np.newaxis

train_xor = True
train_mnist = True

if train_xor:
    D,N = 2,200000

    #this is the XOR problem.
    X = np.random.rand(N,D) #we want [NxD] data
    X = (X > 0.5)*1.0
    Y = X[:,0] == X[:,1]
    Y = (np.vstack((Y, np.invert(Y)))*1.0).T # and [NxC] labels

    X += np.random.randn(N,D)*0.1 # add some noise to the data.

    #build a network
    nn = modules.Sequential([modules.Linear(2,3), modules.Tanh(),modules.Linear(3,15), modules.Tanh(), modules.Linear(15,15), modules.Tanh(), modules.Linear(15,3), modules.Tanh() ,modules.Linear(3,2), modules.SoftMax()])
    #train the network.
    nn.train(X,Y, batchsize = 5, iters=1000)
    acc = np.mean(np.argmax(nn.forward(X), axis=1) == np.argmax(Y, axis=1))
    if not np == numpy: # np=cupy
        acc = np.asnumpy(acc)
    print('model train accuracy is: {:0.4f}'.format(acc))

    #save the network
    model_io.write(nn, '../xor_net_small_1000.txt')

if train_mnist:

    Xtrain = data_io.read('../data/MNIST/train_images.npy')
    Ytrain = data_io.read('../data/MNIST/train_labels.npy')
    Xtest = data_io.read('../data/MNIST/test_images.npy')
    Ytest = data_io.read('../data/MNIST/test_labels.npy')

    # transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
    Xtrain =  Xtrain / 127.5 - 1
    Xtest =  Xtest / 127.5 - 1

    # transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
    I = Ytrain[:,0].astype(int)
    Ytrain = np.zeros([Xtrain.shape[0],np.unique(Ytrain).size])
    Ytrain[np.arange(Ytrain.shape[0]),I] = 1

    I = Ytest[:,0].astype(int)
    Ytest = np.zeros([Xtest.shape[0],np.unique(Ytest).size])
    Ytest[np.arange(Ytest.shape[0]),I] = 1

    nn = modules.Sequential(
        [
            modules.Flatten(),
            modules.Linear(784, 1296),
            modules.Rect(),
            modules.Linear(1296,1296),
            modules.Rect(),
            modules.Linear(1296,1296),
            modules.Rect(),
            modules.Linear(1296, 10),
            modules.SoftMax()
        ]
    )
    
    nn.train(Xtrain, Ytrain, Xtest, Ytest, batchsize=64, iters=35000, status=1000)
    acc = np.mean(np.argmax(nn.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
    if not np == numpy: # np=cupy
        acc = np.asnumpy(acc)
    print('model test accuracy is: {:0.4f}'.format(acc))
    model_io.write(nn, '../mnist_mlp-1296-1296-1296.txt')

    #try loading the model again and compute score, see if this checks out. this time in numpy
    nn = model_io.read('../mnist_mlp-1296-1296-1296.txt')
    acc = np.mean(np.argmax(nn.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
    if not np == numpy: acc = np.asnumpy(acc)
    print('model test accuracy (numpy) is: {:0.4f}'.format(acc))
