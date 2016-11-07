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
        # DY is of shape N, Hout, Wout, nfilters

        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        #distribute the gradient (1 * DY) towards across all contributing inputs evenly
        DX = np.zeros_like(self.X)
        for i in xrange(Hout):
            for j in xrange(Wout):
                DX[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += DY[:,i:i+1,j:j+1,:]

        print 'dx sum sumpool', DX.sum()
        return DX


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

        #distribute the gradient towards across all inputs evenly
        #assumes non-zero values for each input, which should be mostly true -> gradient at each input is 1

        Rx = np.zeros(self.X.shape)
        for i in xrange(Hout):
            for j in xrange(Wout):
                x = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations. N,hpool,wpool,D
                xsum = x.sum(axis=(1,2),keepdims=True)
                z = x / ( xsum + ((xsum >= 0)*2-1.)*1e-12 ) #proportional input activations per layer plus some slight numerical stabilization to avoid division by zero

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += z * R[:,i:i+1,j:j+1,:]  #distribute relevance propoprtional to input activations per layer

        return Rx