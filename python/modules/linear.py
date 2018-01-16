'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import numpy as np
from .module import Module
na = np.newaxis

# -------------------------------
# Linear layer
# -------------------------------
class Linear(Module):
    '''
    Linear Layer
    '''

    def __init__(self,m,n):
        '''
        Initiates an instance of a linear computation layer.

        Parameters
        ----------
        m : int
            input dimensionality
        n : int
            output dimensionality

        Returns
        -------
        the newly created object instance
        '''
        Module.__init__(self)
        self.m = m
        self.n = n
        self.B = np.zeros([self.n])
        self.W = np.random.normal(0,1.0*m**(-.5),[self.m,self.n])


    def forward(self,X):
        '''
        Forward-transforms an input X

        Parameters
        ----------

        X : numpy.ndarray
            the input, shaped [N,D], where N is the number of samples and D their dimensionality

        Returns
        -------
        Y : numpy.ndarray
            the transformed data shaped [N,M], with M being the number of output neurons
        '''

        self.X = X
        self.Y = np.dot(X,self.W)+self.B
        return self.Y


    def backward(self,DY):
        '''
        Backward pass through the linear layer, computing the derivative wrt the inputs.
        Ensures a well-conditioned output gradient

        Parameters
        ----------

        DY : numpy.ndarray
            the backpropagated error signal as input, shaped [N,M]

        Returns
        -------

        DX : numpy.ndarray
            the computed output derivative of the error signal wrt X, shaped [N,D]
        '''

        self.dW = np.dot(self.X.T,DY)
        self.dB = DY.sum(axis=0)
        return np.dot(DY,self.W.T)*self.m**.5/self.n**.5


    def update(self, lrate):
        '''
        Update the model weights
        '''
        self.W -= lrate*self.dW/self.m**.5
        self.B -= lrate*self.dB/self.n**.25


    def clean(self):
        '''
        Removes temporarily stored variables from this layer
        '''
        self.X = None
        self.Y = None
        self.dW = None
        self.dB = None




    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        Z = self.W[na,:,:]*self.X[:,:,na] #localized preactivations
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] #preactivations
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)

    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        note that for fully connected layers, this results in a uniform lower layer relevance map.
        '''
        Z = np.ones_like(self.W[na,:,:])
        Zs = Z.sum(axis=1)[:,na,:]
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)

    def _ww_lrp(self,R):
        '''
        LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''
        Z = self.W[na,:,:]**2
        Zs = Z.sum(axis=1)[:,na,:]
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)

    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] # preactivations

        # add slack to denominator. we require sign(0) = 1. since np.sign(0) = 0 would defeat the purpose of the numeric stabilizer we do not use it.
        Zs += epsilon * ((Zs >= 0)*2-1)
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)


    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        beta = 1 - alpha
        Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations

        if not alpha == 0:
            Zp = Z * (Z > 0)
            Zsp = Zp.sum(axis=1)[:,na,:] + (self.B * (self.B > 0))[na,na,:]
            Ralpha = alpha * ((Zp / Zsp) * R[:,na,:]).sum(axis=2)
        else:
            Ralpha = 0

        if not beta == 0:
            Zn = Z * (Z < 0)
            Zsn = Zn.sum(axis=1)[:,na,:] + (self.B * (self.B < 0))[na,na,:]
            Rbeta = beta * ((Zn / Zsn) * R[:,na,:]).sum(axis=2)
        else:
            Rbeta = 0

        return Ralpha + Rbeta