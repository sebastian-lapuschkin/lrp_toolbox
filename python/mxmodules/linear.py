'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@author: Maximilian Kohlbrenner
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import mxnet as mx
from mxnet import nd
from module import Module

# -------------------------------
# Linear layer
# -------------------------------
class Linear(Module):
    '''
    Linear Layer
    '''

    def __init__(self,m,n, ctx=mx.cpu(), dtype='float32'):
        '''
        Initiates an instance of a linear computation layer.

        Parameters
        ----------
        m :     int
                input dimensionality
        n :     int
                output dimensionality
        ctx:    mxnet.context.Context
                device used for all mxnet.ndarray operations
        dtype:  string ('float32' | 'float64')
                dtype used for all mxnet.ndarray operations
                (mxnet default is 'float32', 'float64' supported for easier comparison with numpy)

        Returns
        -------
        the newly created object instance
        '''
        Module.__init__(self)
        self.m = m
        self.n = n

        # context sensitive variables
        self.ctx = ctx
        self.B = nd.zeros([self.n], ctx=ctx, dtype=dtype)
        self.W = nd.random_normal(0,1.0*m**(-.5),[self.m,self.n], ctx=ctx, dtype=dtype)
        self.Y = None
        self.Z = None

        self.dtype=dtype

    def set_context(self, ctx):
        '''
        Change module context and copy ndarrays (if needed)

        Parameters
        ----------
        ctx:    mxnet.context.Context
                device used for all mxnet.ndarray operations
        '''
        self.ctx = ctx
        # copy variables if ctx != variable.context:
        self.W = self.W.as_in_context(ctx)
        self.B = self.B.as_in_context(ctx)
        if not self.Y is None:
            self.Y = self.Y.as_in_context(ctx)
        if not self.Z is None:
            self.Z = self.Z.as_in_context(ctx)

        # new forward pass is needed after context change, reset self.X
        self.X = None


    def forward(self,X,lrp_aware=False):
        '''
        Forward-transforms an input X

        Parameters
        ----------

        X :         mxnet.ndarray.ndarray.NDArray
                    the input, shaped [N,D], where N is the number of samples and D their dimensionality

        lrp_aware : bool
                    controls whether the forward pass is to be computed with awareness for multiple following
                    LRP calls. this will sacrifice speed in the forward pass but will save time if multiple LRP
                    calls will follow for the current X, e.g. wit different parameter settings or for multiple
                    target classes.
        Returns
        -------
        Y : mxnet.ndarray.ndarray.NDArray
            the transformed data shaped [N,M], with M being the number of output neurons
        '''
        self.lrp_aware = lrp_aware
        if self.lrp_aware:
            self.X = X
            self.Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X, axis=2)
            self.Y = nd.sum(self.Z, axis=1) + self.B
        else:
            self.X = X
            self.Y = nd.dot(X,self.W)+self.B

        return self.Y


    def backward(self,DY):
        '''
        Backward pass through the linear layer, computing the derivative wrt the inputs.
        Ensures a well-conditioned output gradient

        Parameters
        ----------

        DY : mxnet.ndarray.ndarray.NDArray
            the backpropagated error signal as input, shaped [N,M]

        Returns
        -------

        DX : mxnet.ndarray.ndarray.NDArray
            the computed output derivative of the error signal wrt X, shaped [N,D]
        '''

        self.dW = nd.dot(self.X.T,DY)
        self.dB = nd.sum(DY, axis=0)
        return nd.dot(DY,self.W.T) *self.m**.5/self.n**.5


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
        self.X  = None
        self.Y  = None
        self.Z  = None
        self.dW = None
        self.dB = None


    def _simple_lrp_slow(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''
        Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X, axis=2) #localized preactivations
        Zs = nd.expand_dims(nd.sum(Z, axis=1), axis=1) + nd.expand_dims(nd.expand_dims(self.B, axis=0), axis=0)
        Zs += 1e-16*((Zs >= 0)*2 - 1.) #add weak default stabilizer to denominator
        return nd.sum((Z / Zs) * nd.expand_dims(R, axis=1), axis=2)


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        # Has the forward pass been computed lrp-aware?
        # This exchanges time spent in the forward pass for lower LRP time
        # and is useful, if e.g. several parameter settings for LRP need to be evaluated
        # for the same input data.
        Zs = self.Y + 1e-16*( (self.Y >= 0) * 2 - 1.) #add weakdefault stabilizer to denominator
        if self.lrp_aware:
            return nd.sum(self.Z * nd.expand_dims(R/Zs, 1), axis=2)
        else:
            Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X, axis=2) #localized preactivations
            return nd.sum(Z * nd.expand_dims(R/Zs, 1), axis=2)



    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        note that for fully connected layers, this results in a uniform lower layer relevance map.
        '''
        Z = nd.ones(nd.expand_dims(self.W, 0).shape, ctx=self.ctx, dtype=self.dtype)
        Zs = nd.expand_dims(nd.sum(Z, axis=1), axis=1)
        return nd.sum(((Z / Zs) * nd.expand_dims(R, axis=1)), axis=2)


    def _ww_lrp(self,R):
        '''
        LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''
        Z = nd.expand_dims(self.W**2, axis=0)
        Zs = nd.expand_dims(nd.sum(Z, axis=1), axis=1)
        return nd.sum(((Z / Zs) * nd.expand_dims(R, axis=1)), axis=2)


    def _epsilon_lrp_slow(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''

        Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X, axis=2) #localized preactivations
        Zs = nd.expand_dims(nd.sum(Z, axis=1), axis=1) + nd.expand_dims(nd.expand_dims(self.B, axis=0), axis=0) # preactivations

        # add slack to denominator. we require sign(0) = 1. since np.sign(0) = 0 would defeat the purpose of the numeric stabilizer we do not use it.
        Zs += epsilon * ((Zs >= 0)*2-1)
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)


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
            return nd.sum(self.Z * nd.expand_dims(R/Zs, 1), axis=2)
        else:
            Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X, axis=2) #localized preactivations
            return nd.sum(Z * nd.expand_dims(R/Zs, 1), axis=2)


    def _alphabeta_lrp_slow(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''
        beta = 1 - alpha
        Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X, axis=2) #localized preactivations

        if not alpha == 0:
            Zp = Z * (Z > 0)
            Zsp = nd.expand_dims(nd.sum(Zp, axis=1), axis=1) + nd.expand_dims(nd.expand_dims(self.B * (self.B > 0), axis=0), axis=0)
            Ralpha = alpha * nd.sum(((Zp / Zsp) * nd.expand_dims(R, axis=1)), axis=2)
        else:
            Ralpha = 0

        if not beta == 0:
            Zn = Z * (Z < 0)
            Zsn = nd.expand_dims(nd.sum(Zn, axis=1), axis=1) + nd.expand_dims(nd.expand_dimx(self.B * (self.B < 0), axis=0), axis=0)
            Rbeta = beta * nd.sum(((Zn / Zsn) * nd.expand_dims(R, axis=1)), axis=2)
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
            Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X, axis=2) #localized preactivations


        #index mask of positive forward predictions
        Zplus = Z > 0
        if alpha * beta != 0: #the general case: both parameters are not 0
            Zp = Z * Zplus
            Zsp = nd.sum(Zp, axis=1) + nd.expand_dims(self.B * (self.B > 0), axis=0) + 1e-16

            Zn = Z - Zp
            Zsn = self.Y - Zsp - 1e-16

            return alpha * nd.sum(Zp * nd.expand_dims(R/Zsp, axis=1), axis=2) + beta * nd.sum(Zn * nd.expand_dims(R/Zsn, axis=1), axis=2)

        elif alpha: #only alpha is not 0 -> alpha = 1, beta = 0
            Zp = Z * Zplus
            Zsp = nd.sum(Zp, axis=1) + nd.expand_dims(self.B * (self.B > 0), axis=0) + 1e-16
            return nd.sum(Zp * nd.expand_dims(R/Zsp, axis=1), axis=2)

        elif beta: # only beta is not 0 -> alpha = 0, beta = 1
            Zn = Z <= 0 # ist this correct? Z * np.invert(Zplus)
            Zsn = nd.sum(Zn, axis=1) + nd.expand_dims(self.B * (self.B < 0), axis=0) - 1e-16
            return nd.sum(Zn * nd.expand_dims(R/Zsn, axis=1), axis=2)

        else:
            raise Exception('This case should never occur: alpha={}, beta={}.'.format(alpha, beta))
