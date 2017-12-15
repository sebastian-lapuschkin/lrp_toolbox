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
# 2D Convolution layer
# -------------------------------

class Convolution(Module):

    def __init__(self, filtersize=(5,5,3,32), stride = (2,2), ctx=mx.cpu()):
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

        '''

        Module.__init__(self)

        self.ctx = ctx

        self.fh, self.fw, self.fd, self.n = filtersize
        self.stride = stride

        self.W = nd.random.normal(0,1./(self.fh*self.fw*self.fd)**.5, shape=filtersize, ctx=ctx)
        self.B = nd.zeros([self.n], ctx=ctx)


    def forward(self,X):
        '''
        Realizes the forward pass of an input through the convolution layer.

        Parameters
        ----------
        X : numpy.ndarray
            a network input, shaped (N,H,W,D), with
            N = batch size
            H, W, D = input size in heigth, width, depth

        Returns
        -------
        Y : numpy.ndarray
            the layer outputs.
        '''

        self.X = X
        N,H,W,D = X.shape

        hf, wf, df, nf  = self.W.shape
        hstride, wstride = self.stride
        numfilters = self.n

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hf) / hstride + 1
        Wout = (W - wf) / wstride + 1

        #initialize pooled output
        self.Y = nd.zeros((N,Hout,Wout,numfilters), ctx=self.ctx)

        for i in xrange(Hout):
            for j in xrange(Wout):
                # numpy version used tensordot, replace by transposition and sum
                # self.Y[:,i,j,:] = np.tensordot(X[:, i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ],self.W,axes = ([1,2,3],[0,1,2])) + self.B
                # np.sum( X[:, i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , :, None].transpose(1,2,3,0,4) *  self.W[:,:,:,None,:], axis= (0,1,2)) + self.B
                self.Y[:,i,j,:] = nd.sum( nd.expand_dims( X[:, i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ].transpose((1,2,3,0)), 4) * nd.expand_dims(self.W, 3), axis=(0,1,2))  + self.B

        return self.Y


    def backward(self,DY):
        '''
        Backward-passes an input error gradient DY towards the input neurons of this layer.

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

        self.DY = DY
        N,Hy,Wy,NF = DY.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        DX = nd.zeros_like(self.X,dtype="float", ctx=self.ctx)


        if not (hf == wf and self.stride == (1,1)):
            for i in xrange(Hy):
                for j in xrange(Wy):
                    DX[:,i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ] += (self.W[na,...] * DY[:,i:i+1,j:j+1,na,:]).sum(axis=4)  #sum over all the filters
        else:
            for i in xrange(hf):
                for j in xrange(wf):
                    DX[:,i:i+Hy:hstride,j:j+Wy:wstride,:] += nd.dot(DY,self.W[i,j,:,:].T)

        return DX #* (hf*wf*df)**.5 / (NF*Hy*Wy)**.5


    def update(self,lrate):
        N,Hx,Wx,Dx = self.X.shape
        N,Hy,Wy,NF = self.DY.shape

        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        DW = nd.zeros_like(self.W,dtype="float", ctx=self.ctx)

        if not (hf == wf and self.stride == (1,1)):
            for i in xrange(Hy):
                for j in xrange(Wy):
                    DW += (self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :, na] * self.DY[:,i:i+1,j:j+1,na,:]).sum(axis=0)
        else:
            for i in xrange(hf):
                for j in xrange(wf):
                    # np tensordot formulation:
                    # DW[i,j,:,:] = np.tensordot(self.X[:,i:i+Hy:hstride,j:j+Wy:wstride,:],self.DY,axes=([0,1,2],[0,1,2]))
                    DW[i,j,:,:] = nd.sum( nd.expand_dims(self.X[:,i:i+Hy:hstride,j:j+Wy:wstride,:], axis=4) * nd.expand_dims(self.DY, axis=3) ,axis=(0,1,2))

        DB = self.DY.sum(axis=(0,1,2))
        self.W -= lrate * DW / (hf*wf*df*Hy*Wy)**.5
        self.B -= lrate * DB / (Hy*Wy)**.5


    def clean(self):
        self.X = None
        self.Y = None
        self.DY = None


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        N,Hout,Wout,NF = R.shape
        hf,wf,df,NF = self.W.shape
        hstride, wstride = self.stride

        Rx = nd.zeros_like(self.X,dtype="float", ctx=self.ctx)

        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = nd.expand_dims(self.W, 0) * nd.expand_dims(self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ], 4)
                Zs = Z.sum(axis=(1,2,3),keepdims=True) + nd.expand_dims(nd.expand_dims(nd.expand_dims(nd.expand_dims(self.B, 0), 0), 0), 0)
                Zs += 1e-12*((Zs >= 0)*2 - 1.) # add a weak numerical stabilizer to cushion division by zero
                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * nd.expand_dims(R[:,i:i+1,j:j+1,:], 3)).sum(axis=4)
        return Rx


    # TODO: fix broadcasting:

    # def _flat_lrp(self,R):
    #     '''
    #     distribute relevance for each output evenly to the output neurons' receptive fields.
    #     '''
    #
    #     N,Hout,Wout,NF = R.shape
    #     hf,wf,df,NF = self.W.shape
    #     hstride, wstride = self.stride
    #
    #     Rx = nd.zeros_like(self.X,dtype="float", ctx = self.ctx)
    #
    #     for i in xrange(Hout):
    #         for j in xrange(Wout):
    #             Z = nd.ones((N,hf,wf,df,NF), ctx=self.ctx)
    #             Zs = Z.sum(axis=(1,2,3),keepdims=True)
    #
    #             Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
    #     return Rx
    #
    # def _ww_lrp(self,R):
    #     '''
    #     LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
    #     '''
    #
    #     N,Hout,Wout,NF = R.shape
    #     hf,wf,df,NF = self.W.shape
    #     hstride, wstride = self.stride
    #
    #     Rx = nd.zeros_like(self.X,dtype="float", ctx = self.ctx)
    #
    #     for i in xrange(Hout):
    #         for j in xrange(Wout):
    #             Z = self.W[na,...]**2
    #             Zs = Z.sum(axis=(1,2,3),keepdims=True)
    #
    #             Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
    #     return Rx
    #
    # def _epsilon_lrp(self,R,epsilon):
    #     '''
    #     LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
    #     '''
    #
    #     N,Hout,Wout,NF = R.shape
    #     hf,wf,df,NF = self.W.shape
    #     hstride, wstride = self.stride
    #
    #     Rx = nd.zeros_like(self.X,dtype="float", ctx=self.ctx)
    #
    #     for i in xrange(Hout):
    #         for j in xrange(Wout):
    #             Z = self.W[na,...] * self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , na]
    #             Zs = Z.sum(axis=(1,2,3),keepdims=True) + self.B[na,na,na,na,...]
    #             Zs += epsilon*((Zs >= 0)*2-1)
    #             Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += ((Z/Zs) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
    #     return Rx
    #
    #
    # def _alphabeta_lrp(self,R,alpha):
    #     '''
    #     LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
    #     '''
    #
    #     beta = 1 - alpha
    #
    #     N,Hout,Wout,NF = R.shape
    #     hf,wf,df,NF = self.W.shape
    #     hstride, wstride = self.stride
    #
    #     Rx = nd.zeros_like(self.X,dtype="float", ctx = self.ctx)
    #
    #     for i in xrange(Hout):
    #         for j in xrange(Wout):
    #             Z = self.W[na,...] * self.X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , na]
    #
    #             if not alpha == 0:
    #                 Zp = Z * (Z > 0)
    #                 Bp = (self.B * (self.B > 0))[na,na,na,na,...]
    #                 Zsp = Zp.sum(axis=(1,2,3),keepdims=True) + Bp
    #                 Ralpha = alpha * ((Zp/Zsp) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
    #             else:
    #                 Ralpha = 0
    #
    #             if not beta == 0:
    #                 Zn = Z * (Z < 0)
    #                 Bn = (self.B * (self.B < 0))[na,na,na,na,...]
    #                 Zsn = Zn.sum(axis=(1,2,3),keepdims=True) + Bn
    #                 Rbeta = beta * ((Zn/Zsn) * R[:,i:i+1,j:j+1,na,:]).sum(axis=4)
    #             else:
    #                 Rbeta = 0
    #
    #             Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += Ralpha + Rbeta
    #
    #     return Rx
