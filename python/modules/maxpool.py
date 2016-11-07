'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
@license : BSD-2-Clause
'''

import numpy as np
from module import Module

# -------------------------------
# Max Pooling layer
# -------------------------------

class MaxPool(Module):
    def __init__(self,pool=(2,2),stride=(2,2)):
        self.pool = pool
        self.stride = stride


    def forward(self,X):
        '''
        Realizes the forward pass of an input through the max pooling layer.

        Parameters
        ----------
        X : numpy.ndarray
            a network input, shaped (N,H,W,D), with
            N = batch size
            H, W, D = input size in heigth, width, depth

        Returns
        -------
        Y : numpy.ndarray
            the max-pooled outputs, reduced in size due to given stride and pooling size
        '''

        self.X = X
        N,H,W,D = X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        #initialize pooled output
        self.Y = np.zeros((N,Hout,Wout,D))

        for i in xrange(Hout):
            for j in xrange(Wout):
                x = X[:, i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ]
                self.Y[:,i,j,:] = np.amax(np.amax(x,axis=2),axis=1)

        return self.Y


    def backward(self,DY):

        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        #distribute the gradient towards the max activation (evenly in case of ambiguities)
        #the max activation value is already known via self.Y

        #this implementation seems wasteful.....
        DYout = np.zeros_like(self.X,dtype=np.float)
        for i in xrange(Hout):
            for j in xrange(Wout):
                y = self.Y[:,i,j,:] #y is the max activation for this poolig OP. outputs a 2-axis array. over N and D
                x = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations

                #attribute gradient weights per input depth and sample
                for n in xrange(N):
                    for d in xrange(D):
                        activators = x[n,...,d] == y[n,d]
                        DYout[n,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool, d ] += activators * DY[n,i,j,d]
        return DYout


    def clean(self):
        self.X = None
        self.Y = None




    def lrp(self,R, *args, **kwargs):
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        #distribute the gradient towards the max activation (evenly in case of ambiguities)
        #the max activation value is already known via self.Y

        #this implementation seems wasteful.....
        Rx = np.zeros_like(self.X,dtype=np.float)
        for i in xrange(Hout):
            for j in xrange(Wout):
                y = self.Y[:,i,j,:] #y is the max activation for this poolig OP. outputs a 2-axis array. over N and D
                x = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations

                #attribute gradient weights per input depth and sample
                for n in xrange(N):
                    for d in xrange(D):
                        activators = x[n,...,d] == y[n,d]
                        Rx[n,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool, d ] += (activators * DY[n,i,j,d]) * (1./activators.sum()) #last bit to distribute gradient evenly in case of multiple activations.
        return Rx



