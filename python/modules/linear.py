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

        self.m = m
        self.n = n
        self.B = np.zeros([self.n])
        self.W = np.random.normal(0,1.0*m**(-.5),[self.m,self.n])


    def forward(self,X):
        self.X = X
        self.Y = np.dot(X,self.W)+self.B
        return self.Y


    def backward(self,DY):
        self.dW = np.dot(self.X.T,DY)
        self.dB = DY.sum(axis=0)
        return np.dot(DY,self.W.T)


    def update(self, lrate):
        self.W -= lrate*self.dW
        self.B -= lrate*self.dB


    def clean(self):
        self.X = None
        self.Y = None
        self.dW = None
        self.dB = None




    def lrp(self,R, lrp_var=None,param=0):
        '''
        performs LRP by calling subroutines, depending on lrp_var and param

        Parameters
        ----------

        R : numpy.ndarray
            relevance input for LRP.
            should be of the same shape as the previously produced output by Linear.forward

        lrp_var : str
            either 'none' or 'simple' or None for standard Lrp ,
            'epsilon' for an added epsilon slack in the denominator
            'alphabeta' for weighting positive and negative contributions separately. param specifies alpha with alpha + beat = 1

        param : double
            the respective parameter for the lrp method of choice

        Returns
        -------
        R : the backward-propagated relevance scores.
            shaped identically to the previously processed inputs in Linear.forward
        '''

        if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
            return self._simple_lrp(R)
        elif lrp_var.lower() == 'epsilon':
            return self._epsilon_lrp(R,param)
        elif lrp_var.lower() == 'alphabeta':
            return self._alphabeta_lrp(R,param)
        else:
            print 'Unknown lrp variant', lrp_var


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        Z = self.W[na,:,:]*self.X[:,:,na] #localized preactivations
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:] #preactivations
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