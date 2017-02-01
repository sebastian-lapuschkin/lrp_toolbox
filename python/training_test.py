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

import numpy as np ; na = np.newaxis

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
nn.train(X,Y,Xval=X,Yval=Y, batchsize = 5)

#save the network
model_io.write(nn, '../xor_net_small_1000.txt')



