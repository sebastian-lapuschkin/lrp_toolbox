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

from .module import Module
import numpy
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"):
    import cupy
    import cupy as np
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


    def to_cupy(self):
        global np
        assert imp.find_spec("cupy"), "module cupy not found."
        self.W = cupy.array(self.W)
        self.B = cupy.array(self.B)
        if hasattr(self, 'X') and self.X is not None: self.X = cupy.array(self.X)
        if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.array(self.Y)
        if hasattr(self, 'Z') and self.Z is not None: self.Z = cupy.array(self.Z)
        if hasattr(self, 'dW') and self.dW is not None: self.dW = cupy.array(self.dW)
        if hasattr(self, 'dB') and self.dB is not None: self.dB = cupy.array(self.dB)
        np = cupy # ensure correct numerics backend

    def to_numpy(self):
        global np
        if not imp.find_spec("cupy"):
            pass #nothing to do if there is no cupy. model should exist as numpy arrays
        else:
            self.W = cupy.asnumpy(self.W)
            self.B = cupy.asnumpy(self.B)
            if hasattr(self, 'X') and self.X is not None: self.X = cupy.asnumpy(self.X)
            if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.asnumpy(self.Y)
            if hasattr(self, 'Z') and self.Z is not None: self.Z = cupy.asnumpy(self.Z)
            if hasattr(self, 'dW') and self.dW is not None: self.dW = cupy.asnumpy(self.dW)
            if hasattr(self, 'dB') and self.dB is not None: self.dB = cupy.asnumpy(self.dB)
            np = numpy # ensure correct numerics backend


    def forward(self,X,lrp_aware=False):
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
        self.lrp_aware = lrp_aware
        if self.lrp_aware:
            self.X = X
            self.Z = self.W[na,:,:]*self.X[:,:,na]
            self.Y = self.Z.sum(axis=1) + self.B
        else:
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
        self.Z = None
        self.dW = None
        self.dB = None


    def _simple_lrp_slow(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140.
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''
        Z = self.W[na,:,:]*self.X[:,:,na] #localized preactivations
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] #preactivations
        Zs += 1e-16*((Zs >= 0)*2 - 1.) #add weak default stabilizer to denominator
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        # Has the forward pass been computed lrp-aware?
        # This exchanges time spent in the forward pass for lower LRP time
        # and is useful, if e.g. several parameter settings for LRP need to be evaluated
        # for the same input data.
        Zs = self.Y + 1e-16*((self.Y >= 0)*2 - 1.) #add weakdefault stabilizer to denominator
        if self.lrp_aware:
            return (self.Z * (R/Zs)[:,na,:]).sum(axis=2)
        else:
            Z = self.W[na,:,:]*self.X[:,:,na] #localized preactivations
            return (Z * (R/Zs)[:,na,:]).sum(axis=2)



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


    def _epsilon_lrp_slow(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''

        Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] # preactivations

        # add slack to denominator. we require sign(0) = 1. since np.sign(0) = 0 would defeat the purpose of the numeric stabilizer we do not use it.
        Zs += epsilon * ((Zs >= 0)*2-1)
        return  ((Z / Zs) * R[:,na,:]).sum(axis=2)


    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''

        Zs = self.Y + epsilon * ((self.Y >= 0)*2-1)#prepare stabilized denominator

        # Has the forward pass been computed lrp-aware?
        # This exchanges time spent in the forward pass for lower LRP time
        # and is useful, if e.g. several parameter settings for LRP need to be evaluated
        # for the same input data.
        if self.lrp_aware:
            return (self.Z * (R/Zs)[:,na,:]).sum(axis=2)
        else:
            Z = self.W[na,:,:]*self.X[:,:,na] #localized preactivations
            return (Z * (R/Zs)[:,na,:]).sum(axis=2)


    def _alphabeta_lrp_slow(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
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


    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        beta = 1 - alpha
        if self.lrp_aware:
            Z = self.Z
        else:
            Z = self.W[na,:,:]*self.X[:,:,na] # localized preactivations


        #index mask of positive forward predictions
        Zplus = Z > 0
        if alpha * beta != 0: #the general case: both parameters are not 0
            Zp = Z * Zplus
            Zsp = Zp.sum(axis=1) + (self.B * (self.B > 0))[na,:] + 1e-16

            Zn = Z - Zp
            Zsn = self.Y - Zsp - 1e-16

            return alpha * (Zp*(R/Zsp)[:,na,:]).sum(axis=2) + beta * (Zn * (R/Zsn)[:,na,:]).sum(axis=2)

        elif alpha: #only alpha is not 0 -> alpha = 1, beta = 0
            Zp = Z * Zplus
            Zsp = Zp.sum(axis=1) + (self.B * (self.B > 0))[na,:] + 1e-16
            return (Zp*(R/Zsp)[:,na,:]).sum(axis=2)

        elif beta: # only beta is not 0 -> alpha = 0, beta = 1
            Zn = Z * np.invert(Zplus)
            Zsn = Zn.sum(axis=1) + (self.B * (self.B < 0))[na,:] - 1e-16
            return (Zn * (R/Zsn)[:,na,:]).sum(axis=2)

        else:
            raise Exception('This case should never occur: alpha={}, beta={}.'.format(alpha, beta))

