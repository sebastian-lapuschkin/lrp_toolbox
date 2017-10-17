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

import copy
import sys
import numpy as np
import time
from module import Module
na = np.newaxis

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
        Module.__init__(self)
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


    def backward(self,DY):
        for m in self.modules[::-1]:
            DY = m.backward(DY)
        return DY


    def update(self,lrate):
        for m in self.modules:
            m.update(lrate)


    def clean(self):
        '''
        Removes temporary variables from all network layers.
        '''
        for m in self.modules:
            m.clean()


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
                range of indices [0,N[ to choose from.m, s = divmod(seconds, 60)


            b : the number of unique indices to pick.
            '''
            assert(b <= N) # if this fails no valid solution can be found.
            I = np.arange(0)
            while I.size < b:
                I = np.unique(np.append(I,np.random.randint(0,N,[b-I.size,])))

            return I

        t_start = time.time()
        untilConvergence = convergence;    learningFactor = lfactor_initial
        bestAccuracy = 0.0;                bestLayers = copy.deepcopy(self.modules)
        bestLoss = np.Inf;                 bestIter = 0

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
            self.backward(Ypred - Y[samples,:]) #l1-loss
            self.update(lrate*learningFactor)

            #periodically evaluate network and optionally adjust learning rate or check for convergence.
            if (d+1) % status == 0:
                if not Xval == [] and not Yval == []: #if given, evaluate on validation data
                    Ypred = self.forward(Xval)
                    acc = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Yval, axis=1))
                    l1loss = np.abs(Ypred - Yval).sum()/Yval.shape[0]
                    print 'Accuracy after {0} iterations on validation set: {1}% (l1-loss: {2:.4})'.format(d+1, acc*100,l1loss)

                else: #evaluate on the training data only
                    Ypred = self.forward(X)
                    acc = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Y, axis=1))
                    l1loss = np.abs(Ypred - Y).sum()/Y.shape[0]
                    print 'Accuracy after {0} iterations on training data: {1}% (l1-loss: {2:.4})'.format(d+1,acc*100,l1loss)


                #save current network parameters if we have improved
                #if acc >= bestAccuracy and l1loss <= bestLoss:
                # only go by loss
                if l1loss <= bestLoss:
                    print '    New loss-optimal parameter set encountered. saving....'
                    bestAccuracy = acc
                    bestLoss = l1loss
                    bestLayers = copy.deepcopy(self.modules)
                    bestIter = d

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

                t_elapsed =  time.time() - t_start
                percent_done = float(d+1)/iters #d+1 because we are after the iteration's heavy lifting
                t_remaining_estimated = t_elapsed/percent_done - t_elapsed

                m, s = divmod(t_remaining_estimated, 60)
                h, m = divmod(m, 60)
                d, h = divmod(h, 24)

                timestring = '{}d {}h {}m {}s'.format(int(d), int(h), int(m), int(s))
                print '    Estimate time until current training ends : {} ({:.2f}% done)'.format(timestring, percent_done*100)

            elif (d+1) % (status/10) == 0:
                # print 'alive' signal
                #sys.stdout.write('.')
                l1loss = np.abs(Ypred - Y[samples,:]).sum()/Ypred.shape[0]
                sys.stdout.write('batch# {}, lrate {}, l1-loss {:.4}\n'.format(d+1,lrate*learningFactor,l1loss))
                sys.stdout.flush()

        #after training, either due to convergence or iteration limit
        print 'Setting network parameters to best encountered network state with {}% accuracy and a loss of {} from iteration {}.'.format(bestAccuracy*100, bestLoss, bestIter)
        self.modules = bestLayers


    def set_lrp_parameters(self,lrp_var=None,param=None):
        for m in self.modules:
            m.set_lrp_parameters(lrp_var=lrp_var,param=param)

    def lrp(self,R,lrp_var=None,param=None):
        '''
        Performs LRP by calling subroutines, depending on lrp_var and param or
        preset values specified via Module.set_lrp_parameters(lrp_var,lrp_param)

        If lrp parameters have been pre-specified (per layer), the corresponding decomposition
        will be applied during a call of lrp().

        Specifying lrp parameters explicitly when calling lrp(), e.g. net.lrp(R,lrp_var='alpha',param=2.),
        will override the preset values for the current call.

        How to use:

        net.forward(X) #forward feed some data you wish to explain to populat the net.

        then either:

        net.lrp() #to perform the naive approach to lrp implemented in _simple_lrp for each layer

        or:

        for m in net.modules:
            m.set_lrp_parameters(...)
        net.lrp() #to preset a lrp configuration to each layer in the net

        or:

        net.lrp(somevariantname,someparameter) # to explicitly call the specified parametrization for all layers (where applicable) and override any preset configurations.

        Parameters
        ----------
        R : numpy.ndarray
            final layer relevance values. usually the network's prediction of some data points
            for which the output relevance is to be computed
            dimensionality should be equal to the previously computed predictions

        lrp_var : str
            either 'none' or 'simple' or None for standard Lrp ,
            'epsilon' for an added epsilon slack in the denominator
            'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
            'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
            'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.

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
