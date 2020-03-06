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
# Sum Pooling layer
# -------------------------------

class SumPool(Module):

    def __init__(self,pool=(2,2),stride=(2,2)):
        '''
        Constructor for the sum pooling layer object

        Parameters
        ----------

        pool : tuple (h,w)
            the size of the pooling mask in vertical (h) and horizontal (w) direction

        stride : tuple (h,w)
            the vertical (h) and horizontal (w) step sizes between filter applications.
        '''

        Module.__init__(self)

        self.pool = pool
        self.stride = stride

    def to_cupy(self):
        global np
        assert imp.find_spec("cupy"), "module cupy not found."
        if hasattr(self, 'X') and self.X is not None: self.X = cupy.array(self.X)
        if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.array(self.Y)
        np = cupy

    def to_numpy(self):
        global np
        if not imp.find_spec("cupy"):
            pass #nothing to do if there is no cupy. model should exist as numpy arrays
        else:
            if hasattr(self, 'X') and self.X is not None: self.X = cupy.asnumpy(self.X)
            if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.asnumpy(self.Y)
        np = numpy


    def forward(self,X,*args,**kwargs):
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
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        normalizer = 1./np.sqrt(hpool*wpool)

        #initialize pooled output
        self.Y = np.zeros((N,Hout,Wout,D))

        for i in range(Hout):
            for j in range(Wout):
                self.Y[:,i,j,:] = X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ].sum(axis=(1,2)) * normalizer #normalizer to keep the output well conditioned
        return self.Y


    def backward(self,DY):
        '''
        Backward-passes an input error gradient DY towards the input neurons of this sum pooling layer.

        Parameters
        ----------

        DY : numpy.ndarray
            an error gradient shaped same as the output array of forward, i.e. (N,Hy,Wy,Dy) with
            N = number of samples in the batch
            Hy = heigth of the output
            Wy = width of the output
            Dy = output depth = input depth


        Returns
        -------

        DX : numpy.ndarray
            the error gradient propagated towards the input

        '''

        # DY is of shape N, Hout, Wout, nfilters
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        normalizer = 1./np.sqrt(hpool * wpool)

        #distribute the gradient (1 * DY) towards across all contributing inputs evenly
        DX = np.zeros_like(self.X)
        for i in range(Hout):
            for j in range(Wout):
                DX[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += DY[:,i:i+1,j:j+1,:] * normalizer # normalizer to produce well-conditioned gradients
        return DX


    def clean(self):
        self.X = None
        self.Y = None

    def _simple_lrp_slow(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        Rx = np.zeros(self.X.shape)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.
                Zs = Z.sum(axis=(1,2),keepdims=True)
                Zs += 1e-12*((Zs >= 0)*2-1) # add a weak numerical stabilizer to cushion an all-zero input

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += (Z/Zs) * R[:,i:i+1,j:j+1,:]  #distribute relevance propoprtional to input activations per layer

        return Rx


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        Rx = np.zeros(self.X.shape)
        normalizer = 1./np.sqrt(hpool*wpool)
        R_norm = R / (self.Y/normalizer + 1e-12*((self.Y/normalizer >= 0)*2 - 1.)) #factor in normalizer applied to Y in the forward pass


        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.
                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += Z * (R_norm[:,i:i+1,j:j+1,:])

        return Rx


    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = np.ones([N,hpool,wpool,D])
                Zs = Z.sum(axis=(1,2),keepdims=True)
                Rx[:,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool,:] += (Z / Zs) * R[:,i:i+1,j:j+1,:]
        return Rx

    def _ww_lrp(self,R):
        '''
        due to uniform weights used for sum pooling (1), this method defaults to _flat_lrp(R)
        '''
        return self._flat_lrp(R)


    def _epsilon_lrp_slow(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        Rx = np.zeros(self.X.shape)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.
                Zs = Z.sum(axis=(1,2),keepdims=True)
                Zs += epsilon*((Zs >= 0)*2-1) # add a epsilon stabilizer to cushion an all-zero input

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += (Z/Zs) * R[:,i:i+1,j:j+1,:]  #distribute relevance propoprtional to input activations per layer

        return Rx


    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        Rx = np.zeros(self.X.shape)
        normalizer = 1./np.sqrt(hpool*wpool) #factor in normalizer applied to Y in the forward pass
        R_norm = R / (self.Y/normalizer + epsilon*((self.Y >= 0)*2 - 1.))

        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.
                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += Z * (R_norm[:,i:i+1,j:j+1,:])

        return Rx



    #def _epsilon_lrp(self,R,epsilon):
    #    return self._epsilon_lrp_slow(R,epsilon)





    # yes, we can do this. no, it will not make sense most of the time.  by default, _lrp_simple will be called. see line 152
    def _alphabeta_lrp_slow(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''

        beta = 1-alpha

        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        #distribute the gradient towards across all inputs evenly
        Rx = np.zeros(self.X.shape)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.

                if not alpha == 0:
                    Zp = Z * (Z > 0)
                    Zsp = Zp.sum(axis=(1,2),keepdims=True) + 1e-16 #zero division is quite likely in sum pooling layers when using the alpha-variant
                    Ralpha = alpha * (Zp/Zsp) * R[:,i:i+1,j:j+1,:]
                else:
                    Ralpha = 0

                if not beta == 0:
                    Zn = Z * (Z < 0)
                    Zsn = Zn.sum(axis=(1,2),keepdims=True) - 1e-16 #zero division is quite likely in sum pooling layers when using the alpha-variant
                    Rbeta = beta * (Zn/Zsn) * R[:,i:i+1,j:j+1,:]
                else:
                    Rbeta = 0

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += Ralpha + Rbeta

        return Rx

    def _alphabeta_lrp(self, R, alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''

        beta = 1-alpha

        N,H,W,D = self.X.shape

        hpool,   wpool   = self.pool
        hstride, wstride = self.stride

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)
        normalizer = 1./np.sqrt(hpool*wpool) #factor in normalizer applied to Y in the forward pass

        #distribute the gradient towards across all inputs evenly
        Rx = np.zeros(self.X.shape)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.
                Zplus = Z > 0 #index mask of positive forward predictions

                if alpha * beta != 0 : #the general case: both parameters are not 0
                    Zp = Z * Zplus
                    Zsp = Zp.sum(axis=(1,2),keepdims=True)+ 1e-16

                    Zn = Z - Zp
                    Zsn = self.Y[:,i:i+1,j:j+1,:]/normalizer - Zsp - 1e-16

                    Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += (alpha*(Zp/Zsp) + beta*Zn*(R[:,i:i+1,j:j+1,:]/Zsn))*R[:,i:i+1,j:j+1,:]

                elif alpha: #only alpha is not 0 -> alpha = 1, beta = 0
                    Zp = Z * Zplus
                    Zsp = Zp.sum(axis=(1,2),keepdims=True) + 1e-16
                    Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += Zp*(R[:,i:i+1,j:j+1,:]/Zsp)

                elif beta: # only beta is not 0 -> alpha = 0, beta = 1
                    Zn = Z * np.invert(Zplus)
                    Zsn = Zn.sum(axis=(1,2),keepdims=True) - 1e-16
                    Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += Zn*(R[:,i:i+1,j:j+1,:]/Zsn)

                else:
                    raise Exception('This case should never occur: alpha={}, beta={}.'.format(alpha, beta))

        return Rx
