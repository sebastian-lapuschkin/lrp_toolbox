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

import numpy as np ; na = np.newaxis
import copy
import sys



# -------------------------------
# Modules for the neural network
# -------------------------------
class Module:
    '''
    Superclass for all computation layer implementations
    '''

    def __init__(self): pass
    def update(self, lrate): pass
    def clean(self): pass
    def lrp(self,R, *args, **kwargs): return R
    def backward(self,DY): return DY
    def train(self, X, Y, *args, **kwargs): pass
    def forward(self,X): return X


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

        Zp = Z * (Z > 0)
        Zsp = Zp.sum(axis=1)[:,na,:] + (self.B * (self.B > 0))[na,na,:]

        Zn = Z * (Z < 0)
        Zsn = Zn.sum(axis=1)[:,na,:] + (self.B * (self.B < 0))[na,na,:]

        return alpha * ((Zp / Zsp) * R[:,na,:]).sum(axis=2) + beta * ((Zn / Zsn) * R[:,na,:]).sum(axis=2)


    def backward(self,DY):
        self.dW = np.dot(self.X.T,DY)
        self.dB = DY.sum(axis=0)
        return np.dot(DY,self.W.T)*self.m**.5/self.n**.5


    def update(self, lrate):
        self.W -= lrate*self.dW/self.m**.5
        self.B -= lrate*self.dB/self.m**.25


    def clean(self):
        self.X = None
        self.Y = None
        self.dW = None
        self.dB = None


# -------------------------------
# Tanh layer
# -------------------------------
class Tanh(Module):
    '''
    Tanh Layer
    '''

    def forward(self,X):
        self.Y = np.tanh(X)
        return self.Y


    def backward(self,DY):
        return DY*(1.0-self.Y**2)


    def clean(self):
        self.Y = None

# -------------------------------
# Rectification layer
# -------------------------------
class Rect(Module):
    '''
    Rectification Layer
    '''

    def forward(self,X):
        self.Y = np.maximum(0,X)
        return self.Y


    def backward(self,DY):
        return DY*(self.Y!=0)


    def clean(self):
        self.Y = None


# -------------------------------
# Softmax layer
# -------------------------------
class SoftMax(Module):
    '''
    Softmax Layer
    '''

    def forward(self,X):
        self.X = X
        self.Y = np.exp(X) / np.exp(X).sum(axis=1)[:,na]
        return self.Y


    def lrp(self,R,*args,**kwargs):
        return R*self.X


    def clean(self):
        self.X = None
        self.Y = None


# -------------------------------
# Max Pooling layer
# -------------------------------

class MaxPooling(Module):
    def __init__(self,pool=(2,2),stride=(2,2)):
        self.pool = pool
        self.stride = stride

    def clean(self):
        self.X = None
        self.Y = None

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



    def lrp(self,R, *args, **kwargs):
        #this should behave exactly the same as backward, just with relevance. so let's use what we already have
        return self.backward(R)


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
                        DYout[n,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool, d ] += (activators * DY[n,i,j,d]) * (1./activators.sum()) #last bit to distribute gradient evenly in case of multiple activations.
        return DYout

# -------------------------------
# Sum Pooling layer
# -------------------------------

class SumPooling(Module):
    def __init__(self,pool=(2,2),stride=(2,2)):
        self.pool = pool
        self.stride = stride

    def clean(self):
        self.X = None
        self.Y = None

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

# -------------------------------
# 2D Convolution layer
# -------------------------------

class Convolution(Module):

    def __init__(self, filtersize=(5,5,3,32), stride = (2,2)):
        '''
        filtersize = (h,w,d,n)
        '''

        self.fh, self.fw, self.fd, self.n = filtersize
        self.stride = stride

        self.W = np.random.normal(0,1/(self.fh,self.fw,self.fd)**.5, filtersize)
        self.B = np.zeros([self.n])


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
        self.Y = np.zeros((N,Hout,Wout,numfilters))

        for i in xrange(Hout):
            for j in xrange(Wout):
                x = X[:, i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ]
                y = np.tensordot(x,self.W,axes = ([1,2,3],[0,1,2]))
                self.Y[:,i,j,:] = y + self.B

        return self.Y


    def backward(self,DY):

        self.DY = DY
        N,Hout,Wout,numfilters = DY.shape

        hf, wf, df, numfilters = self.W.shape
        hstride, wstride = self.stride

        W = self.W[None,...] # extend for N axis in input. is 5-tensor now: N,hf,wf,df,numfilters

        DX = np.zeros_like(self.X,dtype=np.float)

        for i in xrange(Hout):
            for j in xrange(Wout):
                dy = DY[:,i:i+1,j:j+1,None,:] # N,1,1,numfilters, extended to N,1,1,df,numfilters
                DX[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += (W * dy).sum(axis=4)  #sum over all the filters

        return DX


    def lrp(self,R, *args, **kwargs):

        N,Hout,Wout,numfilters = R.shape

        hf,wf,df,numfilters = self.W.shape
        hstride, wstride = self.stride

        W = self.W[None,...] # extend for N axis in input. is 5-tensor now: N,hf,wf,df,numfilters

        Rx = np.zeros_like(self.X,dtype=np.float)

        for i in xrange(Hout):
            for j in xrange(Wout):
                x = X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , None] # N,hf,wf,df,numfilters . extended for numfilters = 1.
                Z = W * x #input activations
                Zsum = Z.sum(axis=(1,2,3),keepdims=True) # sum over filter tensors to proportionally distribute over each filter's input
                Z = Z / Zsum #proportional input activations per filter.
                #STABILIZATION. ADD sign2-fxn

                # might cause relevance increase, sneaking in another axis?
                r = R[:,i:i+1,j:j+1,None,:] #N, 1, 1, numfilters, extended to N,1,1,df,numfilters
                r = (Z * r).sum(axis=4) # N, hf, wf, df  ; df = Dx
                Rx[:,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf: , : ] += r #distribute relevance propoprtional to input activations per filter

        return Rx

    def update(self,lrate):

        N,Hx,Wx,Dx = self.X.shape
        N,Hout,Wout,Dout = self.DY.shape

        hf,wf,df,numfilters = self.W.shape
        hstride, wstride = self.stride

        DW = np.zeros_like(self.W,dtype=np.float) # hf, wf, df, numfilters

        for i in xrange(Hout):
            for j in xrange(Wout):
                x = X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , :] # N,hf,wf,df
                x = x[...,None] # N, hf, wf, df, nf=1

                dy = DY[:,i:i+1,j:j+1,None,:] # N, Hout=1, Wout=1, df=1, nf

                # hf, wf, df, nf
                self.DW += (x * dy).sum(axis=0) # hf, wf, df, nf

        self.DB = self.DY.sum(axis=(0,1,2))


        self.W -= lr * self.DW
        self.B -= lr * self.DB


        def clean(self):
            self.X = None
            self.Y = None
            self.DW = None
            self.DB = None


# -------------------------------
# Flattening Layer
# -------------------------------

class Flatten(Module):

    def __init__(self):
        self.inputshape = []

    def lrp(self,R, *args, **kwargs):
        return np.reshape(R,self.inputshape)

    def backward(self,DY):
        return np.reshape(DY,self.inputshape)

    def forward(self,X):
        self.inputshape = X.shape # N x H x W x D
        return np.reshape(X,[self.inputshape[0],np.prod(self.inputshape[1:])])

# -------------------------------
# Sequential layer
# -------------------------------
class Sequential(Module):
    '''
    Top level access point and incorporation of the neural network implementation.
    Sequential manages a sequence of computational neural network modules and passes
    along in- and outputs.
    '''

    def __init__(self,modules):
        '''
        Constructor

        Parameters
        ----------
        modules : list, tuple, etc. enumerable.
            an enumerable collection of instances of class Module
        '''
        self.modules = modules

    def forward(self,X):
        '''
        Realizes the forward pass of an input through the net

        Parameters
        ----------
        X : numpy.ndarray
            a network input.

        Returns
        -------
        X : numpy.ndarray
            the output of the network's final layer
        '''

        for m in self.modules:
            X = m.forward(X)
        return X



    def lrp(self,R,lrp_var=None,param=0):
        '''
        Performs LRP using the network and temporary data produced by a forward call

        Parameters
        ----------
        R : numpy.ndarray
            final layer relevance values. usually the network's prediction of some data points
            for which the output relevance is to be computed
            dimensionality should be equal to the previously computed predictions

        lrpvar : str
            either 'none' or 'simple' or None for standard Lrp ,
            'epsilon' for an added epsilon slack in the denominator
            'alphabeta' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1

        param : double
            the respective parameter for the lrp method of choice

        Returns
        -------

        R : numpy.ndarray
            the first layer relevances as produced by the neural net wrt to the previously forward
            passed input data. dimensionality is equal to the previously into forward entered input data

        Note
        ----

        Requires the net to be populated with temporary variables, i.e. forward needed to be called with the input
        for which the explanation is to be computed. calling clean in between forward and lrp invalidates the
        temporary data
        '''

        for m in self.modules[::-1]:
            R = m.lrp(R,lrp_var,param)
        return R




    def train(self, X, Y,  Xval = [], Yval = [],  batchsize = 25, iters = 10000, lrate = 0.005, lrate_decay = None, lfactor_initial=1.0 , status = 250, convergence = -1, transform = None):
        '''
        Provides a method for training the neural net (self) based on given data.

        Parameters
        ----------

        X : numpy.ndarray
            the training data, formatted to (N,D) shape, with N being the number of samples and D their dimensionality

        Y : numpy.ndarray
            the training labels, formatted to (N,C) shape, with N being the number of samples and C the number of output classes.

        Xval : numpy.ndarray
            some optional validation data. used to measure network performance during training.
            shaped (M,D)

        Yval : numpy.ndarray
            the validation labels. shaped (M,C)

        batchsize : int
            the batch size to use for training

        iters : int
            max number of training iterations

        lrate : float
            the initial learning rate. the learning rate is adjusted during training with increased model performance. See lrate_decay

        lrate_decay : string
            controls if and how the learning rate is adjusted throughout training:
            'none' or None disables learning rate adaption. This is the DEFAULT behaviour.
            'sublinear' adjusts the learning rate to lrate*(1-Accuracy**2) during an evaluation step, often resulting in a better performing model.
            'linear' adjusts the learning rate to lrate*(1-Accuracy) during an evaluation step, often resulting in a better performing model.

        lfactor_initial : float
            specifies an initial discount on the given learning rate, e.g. when retraining an established network in combination with a learning rate decay,
            it might be undesirable to use the given learning rate in the beginning. this could have been done better. TODO: do better.
            Default value is 1.0

        status : int
            number of iterations (i.e. number of rounds of batch forward pass, gradient backward pass, parameter update) of silent training
            until status print and evaluation on validation data.

        convergence : int
            number of consecutive allowed status evaluations with no more model improvements until we accept the model has converged.
            Set <=0 to disable. Disabled by DEFAULT.
            Set to any value > 0 to control the maximal consecutive number (status * convergence) iterations allowed without model improvement, until convergence is accepted.

        transform : function handle
            a function taking as an input a batch of training data sized [N,D] and returning a batch sized [N,D] with added noise or other various data transformations. It's up to you!
            default value is None for no transformation.
            expected syntax is, with X.shape == Xt.shape == (N,D)
            def yourFunction(X):
                Xt = someStuff(X)
                return Xt
        '''

        def randperm(N,b):
            '''
            helper method for picking b unique random indices from a range [0,N[.
            we do not use numpy.random.permutation or numpy.random.choice
            due to known severe performance issues with drawing without replacement.
            if the ratio of N/b is high enough, we should see a huge performance gain.

            N : int
                range of indices [0,N[ to choose from.

            b : the number of unique indices to pick.
            '''
            assert(b <= N) # if this fails no valid solution can be found.
            I = np.arange(0)
            while I.size < b:
                I = np.unique(np.append(I,np.random.randint(0,N,[b-I.size,])))

            return I




        untilConvergence = convergence;    learningFactor = lfactor_initial
        bestAccuracy = 0.0;                bestLayers = copy.deepcopy(self.modules)

        N = X.shape[0]
        for d in xrange(iters):

            #the actual training:
            #first, pick samples at random
            samples = randperm(N,batchsize)

            #transform batch data (maybe)
            if transform == None:
                batch = X[samples,:]
            else:
                batch = transform(X[samples,:])

            #forward and backward propagation steps with parameter update
            Ypred = self.forward(batch)
            self.backward(Ypred - Y[samples,:])
            self.update(lrate*learningFactor)

            #periodically evaluate network and optionally adjust learning rate or check for convergence.
            if (d+1) % status == 0:
                Ypred = self.forward(X)
                acc = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Y, axis=1))
                print
                print 'Accuracy after {0} iterations: {1}%'.format(d+1,acc*100)

                #if given, also evaluate on validation data
                if not Xval == [] and not Yval == []:
                    Ypred = self.forward(Xval)
                    acc_val = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Yval, axis=1))
                    print 'Accuracy on validation set: {1}%'.format(acc_val*100)

                #save current network parameters if we have improved
                if acc > bestAccuracy:
                    print '    New optimal parameter set encountered. saving....'
                    bestAccuracy = acc
                    bestLayers = copy.deepcopy(self.modules)

                    #adjust learning rate
                    if lrate_decay == None or lrate_decay == 'none':
                        pass # no adjustment
                    elif lrate_decay == 'sublinear':
                        #slow down learning to better converge towards an optimum with increased network performance.
                        learningFactor = 1.-(acc*acc)
                        print '    Adjusting learning rate to {0} ~ {1}% of its initial value'.format(learningFactor*lrate, np.round(learningFactor*100,2))
                    elif lrate_decay == 'linear':
                        #slow down learning to better converge towards an optimum with increased network performance.
                        learningFactor = 1.-acc
                        print '    Adjusting learning rate to {0} ~ {1}% of its initial value'.format(learningFactor*lrate, np.round(learningFactor*100,2))

                    #refresh number of allowed search steps until convergence
                    untilConvergence = convergence
                else:
                    untilConvergence-=1
                    if untilConvergence == 0 and convergence > 0:
                        print '    No more recorded model improvements for {0} evaluations. Accepting model convergence.'.format(convergence)
                        break

            elif (d+1) % (status/10) == 0:
                # print 'alive' signal
                sys.stdout.write('.')
                sys.stdout.flush()

        #after training, either due to convergence or iteration limit
        print 'Setting network parameters to best encountered network state with {0}% accuracy.'.format(bestAccuracy*100)
        self.modules = bestLayers



    def backward(self,DY):
        for m in self.modules[::-1]:
            DY = m.backward(DY)
        return DY

    def update(self,lrate):
        for m in self.modules: m.update(lrate)

    def clean(self):
        '''
        Removes temporary variables from all network layers.
        '''
        for m in self.modules: m.clean()


