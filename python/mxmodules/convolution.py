'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@author: Maximilian Kohlbrenner
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 20.10.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import mxnet as mx
from mxnet import nd
from .module import Module


# -------------------------------
# 2D Convolution layer
# -------------------------------

class Convolution(Module):

    def __init__(self, filtersize=(5,5,3,32), stride = (2,2), ctx=mx.cpu(), dtype='float32'):
        '''
        Constructor for a Convolution layer.

        Parameters
        ----------

        filtersize : 4-tuple with values (h,w,d,n), where
            h = filter heigth
            w = filter width
            d = filter depth
            n = number of filters = number of outputs

        stride : 2-tuple (h,w), where
            h = step size for filter application in vertical direction
            w = step size in horizontal direction

        ctx:    mxnet.context.Context
                device used for all mxnet.ndarray operations

        dtype:  string ('float32' | 'float64')
                dtype used for all mxnet.ndarray operations
                (mxnet default is 'float32', 'float64' supported for easier comparison with numpy)
        '''

        Module.__init__(self)

        self.fh, self.fw, self.fd, self.n = filtersize
        self.stride = stride

        # context sensitive variables
        self.ctx = ctx
        self.W = nd.random.normal(0,1./(self.fh*self.fw*self.fd)**.5, shape=filtersize, ctx=ctx, dtype=dtype)
        self.B = nd.zeros([self.n], ctx=ctx, dtype=dtype)
        self.Y = None
        self.Z = None

        # precision:
        self.dtype = dtype

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
        Realizes the forward pass of an input through the convolution layer.

        Parameters
        ----------
        X :         mxnet.ndarray.ndarray.NDArray
                    a network input, shaped (N,H,W,D), with
                    N = batch size
                    H, W, D = input size in heigth, width, depth

        lrp_aware : bool
                    controls whether the forward pass is to be computed with awareness for multiple following
                    LRP calls. this will sacrifice speed in the forward pass but will save time if multiple LRP
                    calls will follow for the current X, e.g. wit different parameter settings or for multiple
                    target classes.

        Returns
        -------
        Y :         mxnet.ndarray.ndarray.NDArray
                    the layer outputs.
        '''

        self.lrp_aware = lrp_aware

        self.X = X
        N,H,W,D = X.shape

        hf, wf, df, nf  = self.W.shape
        hstride, wstride = self.stride
        numfilters = self.n

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hf) // hstride + 1
        Wout = (W - wf) // wstride + 1


        #initialize pooled output
        self.Y = nd.zeros((N,Hout,Wout,numfilters), ctx=self.ctx, dtype=self.dtype)

        if self.lrp_aware:
            self.Z = nd.zeros((N, Hout, Wout, hf, wf, df, nf), ctx=self.ctx, dtype=self.dtype) #initialize container for precomputed forward messages
            for i in range(Hout):
                for j in range(Wout):
                    self.Z[:,i,j,...] = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :], axis=4) # N, hf, wf, df, nf
                    self.Y[:,i,j,:] = self.Z[:,i,j,...].sum(axis=(1,2,3)) + self.B
        else:
            for i in range(Hout):
                for j in range(Wout):
                    self.Y[:,i,j,:] = nd.sum( nd.expand_dims( X[:, i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ].transpose((1,2,3,0)), 4) * nd.expand_dims(self.W, 3), axis=(0,1,2))  + self.B

        return self.Y


    def backward(self,DY):
        '''
        Backward-passes an input error gradient DY towards the input neurons of this layer.

        Parameters
        ----------

        DY :    mxnet.ndarray.ndarray.NDArray
                an error gradient shaped same as the output array of forward, i.e. (N,Hy,Wy,Dy) with
                N = number of samples in the batch
                Hy = heigth of the output
                Wy = width of the output
                Dy = output depth = input depth


        Returns
        -------

        DX :    mxnet.ndarray.ndarray.NDArray
                the error gradient propagated towards the input

        '''

        self.DY = DY
        N,Hy,Wy,NF = DY.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        DX = nd.zeros_like(self.X,ctx=self.ctx, dtype=self.dtype)


        if not (hf == wf and self.stride == (1,1)):
            for i in range(Hy):
                for j in range(Wy):
                    DX[:,i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ] += ( nd.expand_dims(self.W, axis=0) * nd.expand_dims(DY[:,i:i+1,j:j+1,:], axis=3) ).sum(axis=4)  #sum over all the filters
        else:
            for i in range(hf):
                for j in range(wf):
                    DX[:,i:i+Hy:hstride,j:j+Wy:wstride,:] += nd.dot(DY,self.W[i,j,:,:].T)

        return DX #* (hf*wf*df)**.5 / (NF*Hy*Wy)**.5


    def update(self,lrate):
        N,Hx,Wx,Dx = self.X.shape
        N,Hy,Wy,NF = self.DY.shape

        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        DW = nd.zeros_like(self.W,ctx=self.ctx, dtype=self.dtype)

        if not (hf == wf and self.stride == (1,1)):
            for i in range(Hy):
                for j in range(Wy):
                    DW += ( nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :], axis=4) * nd.expand_dims(self.DY[:,i:i+1,j:j+1,:], axis=3)).sum(axis=0)
        else:
            for i in range(hf):
                for j in range(wf):
                    DW[i,j,:,:] = nd.sum( nd.expand_dims(self.X[:,i:i+Hy:hstride,j:j+Wy:wstride,:], axis=4) * nd.expand_dims(self.DY, axis=3) ,axis=(0,1,2))

        DB = self.DY.sum(axis=(0,1,2))
        self.W -= lrate * DW / (hf*wf*df*Hy*Wy)**.5
        self.B -= lrate * DB / (Hy*Wy)**.5


    def clean(self):
        self.X  = None
        self.Y  = None
        self.Z  = None
        self.DY = None


    def _simple_lrp_slow(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx=self.ctx, dtype=self.dtype)

        for i in range(Hout):
            for j in range(Wout):
                Z = nd.expand_dims(self.W, 0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ], 4)
                Zs = Z.sum(axis=(1,2,3),keepdims=True) + nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B, 0), 0), 0), 0)
                Zs += 1e-16*((Zs >= 0)*2 - 1.) # add a weak numerical stabilizer to cushion division by zero
                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * nd.expand_dims(R[:,i:i+1,j:j+1,:], 3)).sum(axis=4)
        return Rx



    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx=self.ctx, dtype=self.dtype)
        R_norm = R / (self.Y + 1e-16*((self.Y >= 0)*2 - 1.))

        for i in range(Hout):
            for j in range(Wout):
                if self.lrp_aware:
                    Z = self.Z[:,i,j,...]
                else:
                    Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :], axis=4)

                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += (Z * nd.expand_dims(R_norm[:,i:i+1,j:j+1,:], 3) ).sum(axis=4)
        return Rx


    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx = self.ctx)

        for i in range(Hout):
            for j in range(Wout):
                Z = nd.ones((N,hf,wf,df,NF), ctx=self.ctx, dtype=self.dtype)
                Zs = Z.sum(axis=(1,2,3),keepdims=True)

                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3) ).sum(axis=4)
        return Rx


    def _ww_lrp(self,R):
        '''
        LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx = self.ctx)

        for i in range(Hout):
            for j in range(Wout):
                Z = nd.expand_dims(self.W, 0)**2
                Zs = Z.sum(axis=(1,2,3),keepdims=True)

                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3)).sum(axis=4)
        return Rx


    def _epsilon_lrp_slow(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx=self.ctx, dtype=self.dtype)

        for i in range(Hout):
            for j in range(Wout):
                Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :], 4)
                Zs = Z.sum(axis=(1,2,3),keepdims=True) + nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B, 0), 0), 0), 0)
                Zs += epsilon*((Zs >= 0)*2-1)
                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3) ).sum(axis=4)
        return Rx


    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx=self.ctx, dtype=self.dtype)
        R_norm = R / (self.Y + epsilon*((self.Y >= 0)*2 - 1.))

        for i in range(Hout):
            for j in range(Wout):
                if self.lrp_aware:
                    Z = self.Z[:,i,j,...]
                else:
                    Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ], axis=4)
                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += (Z * ( nd.expand_dims(R_norm[:,i:i+1,j:j+1,:], axis=3) )).sum(axis=4)
        return Rx


    def _alphabeta_lrp_slow(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        This function shows all necessary operations to perform LRP in one place and is therefore not optimized
        '''

        beta = 1 - alpha

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx = self.ctx)

        for i in range(Hout):
            for j in range(Wout):
                Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :], axis=4)

                if not alpha == 0:
                    Zp = Z * (Z > 0)
                    Bp = nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B * (self.B > 0), axis=0), axis=0), axis=0), axis=0)
                    Zsp = Zp.sum(axis=(1,2,3),keepdims=True) + Bp
                    Ralpha = alpha * ((Zp/Zsp) *  nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3) ).sum(axis=4)
                else:
                    Ralpha = 0

                if not beta == 0:
                    Zn = Z * (Z < 0)
                    Bn = nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B * (self.B < 0), axis=0), axis=0), axis=0), axis=0)
                    Zsn = Zn.sum(axis=(1,2,3),keepdims=True) + Bn
                    Rbeta = beta * ((Zn/Zsn) * nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3) ).sum(axis=4)
                else:
                    Rbeta = 0

                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += Ralpha + Rbeta

        return Rx


    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''

        beta = 1 - alpha

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,ctx = self.ctx)

        for i in range(Hout):
            for j in range(Wout):
                if self.lrp_aware:
                    Z = self.Z[:,i,j,...]
                else:
                    Z = nd.expand_dims(self.W, axis=0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :], axis=4)

                Zplus = Z > 0 #index mask of positive forward predictions

                if alpha * beta != 0 : #the general case: both parameters are not 0
                    Zp = Z * Zplus
                    Zsp = Zp.sum(axis=(1,2,3),keepdims=True) + nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B * (self.B > 0), axis=0), axis=0), axis=0), axis=0) + 1e-16

                    Zn = Z - Zp
                    Zsn = nd.expand_dims(self.Y[:,i:i+1,j:j+1,:], axis=3) - Zsp - 1e-16

                    Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((alpha * (Zp/Zsp) + beta * (Zn/Zsn))* nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3)).sum(axis=4)

                elif alpha: #only alpha is not 0 -> alpha = 1, beta = 0
                    Zp = Z * Zplus
                    Zsp = Zp.sum(axis=(1,2,3),keepdims=True) + nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B * (self.B > 0), axis=0), axis=0), axis=0), axis=0) + 1e-16
                    Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += (Zp*( nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3) /Zsp)).sum(axis=4)

                elif beta: # only beta is not 0 -> alpha = 0, beta = 1
                    Zn = Z * (Z < 0)
                    Zsn = Zn.sum(axis=(1,2,3),keepdims=True) + nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B * (self.B < 0), axis=0), axis=0), axis=0), axis=0) + 1e-16
                    Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += (Zn*( nd.expand_dims(R[:,i:i+1,j:j+1,:], axis=3) /Zsn)).sum(axis=4)

                else:
                    raise Exception('This case should never occur: alpha={}, beta={}.'.format(alpha, beta))

        return Rx
