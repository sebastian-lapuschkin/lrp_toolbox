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
# Sum Pooling layer
# -------------------------------

class SumPool(Module):

    def __init__(self,pool=(2,2),stride=(2,2)):
        self.pool = pool
        self.stride = stride


    def forward(self,X):
        '''
        Realizes the forward pass of an input through the sum pooling layer.

        Parameters
        ----------
        X : numpy.ndarray
            a network input, shaped (N,H,W,D), with
            N = batch size
            H, W, D = input size in heigth, width, depth

        Returns
        -------
        Y : numpy.ndarray
            the sum-pooled outputs, reduced in size due to given stride and pooling size
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
                self.Y[:,i,j,:] = X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ].sum(axis=(1,2))

        return self.Y


    def backward(self,DY):

        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        #distribute the gradient towards across all inputs evenly
        #assumes non-zero values for each input, which should be mostly true -> gradient at each input is 1

        DX = np.zeros_like(self.X)
        for i in xrange(Hout):
            for j in xrange(Wout):
                DX[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += DY[:,i:i+1,j:j+1,:]  #the :: to not lose axis information and allow for broadcasting.
        return DX


    def clean(self):
        self.X = None
        self.Y = None




    def lrp(self,R, *args, **kwargs):

        #copypasta from backward. check for errors if changes made to backward!
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        #distribute the gradient towards across all inputs evenly
        #assumes non-zero values for each input, which should be mostly true -> gradient at each input is 1

        Rx = np.zeros(self.X.shape)
        for i in xrange(Hout):
            for j in xrange(Wout):
                x = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations. N,hpool,wpool,D
                z = x / x.sum(axis=(1,2),keepdims=True) #proportional input activations per layer.
                z[np.isnan(z)] = 1e-12 #do smarter!. isnan is slow.

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += z * R[:,i:i+1,j:j+1,:]  #distribute relevance propoprtional to input activations per layer

        return Rx