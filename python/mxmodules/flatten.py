'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 20.10.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import mxnet as mx
from mxnet import nd
from module import Module

# -------------------------------
# Flattening Layer
# -------------------------------

class Flatten(Module):
    '''
    Flattening layer.
    '''

    def __init__(self):
        Module.__init__(self)
        self.inputshape = []

    def backward(self,DY):
        '''
        Just backward-passes the input gradient DY and reshapes it to fit the input.
        '''
        return nd.reshape(DY,self.inputshape)

    def forward(self,X):
        '''
        Transforms each sample in X to a one-dimensional array.
        Shape change according to C-order.
        '''

        self.inputshape = X.shape # N x H x W x D
        batchsize = self.inputshape[0]
        output = X.reshape((batchsize, X.size / batchsize))

        return output

    def lrp(self,R, *args, **kwargs):
        '''
        Receives upper layer input relevance R and reshapes it to match the input neurons.
        '''
        # just propagate R further down.
        # makes sure subroutines never get called.
        return nd.reshape(R,self.inputshape)
