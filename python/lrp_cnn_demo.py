'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 25.10.2016
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

The purpose of this module is to demonstrate the process of obtaining pixel-wise explanations for given data points at hand of the MNIST hand written digit data set
with CNN models, using the LeNet-5 architecture.

The module first loads a pre-trained neural network model and the MNIST test set with labels and transforms the data such that each pixel value is within the range of [-1 1].
The data is then randomly permuted and for the first 10 samples due to the permuted order, a prediction is computed by the network, which is then as a next step explained
by attributing relevance values to each of the input pixels.

finally, the resulting heatmap is rendered as an image and (over)written out to disk and displayed.
'''


import matplotlib.pyplot as plt
import numpy as np ; na = np.newaxis

import model_io
import data_io
import render

#load a neural network, as well as the MNIST test data and some labels
nn = model_io.read('../models/MNIST/LeNet-5.nn') # 99.23% prediction accuracy
nn.drop_softmax_output_layer() #drop softnax output layer for analyses

X = data_io.read('../data/MNIST/test_images.npy')
Y = data_io.read('../data/MNIST/test_labels.npy')


# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
X =  X / 127.5 - 1.

#reshape the vector representations in X to match the requirements of the CNN input
X = np.reshape(X,[X.shape[0],28,28,1])
X = np.pad(X,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1

acc = np.mean(np.argmax(nn.forward(X), axis=1) == np.argmax(Y, axis=1))
print('model test accuracy is: {:0.4f}'.format(acc))

#permute data order for demonstration. or not. your choice.
I = np.arange(X.shape[0])
#I = np.random.permutation(I)

#predict and perform LRP for the 10 first samples
for i in I[:10]:
    x = X[i:i+1,...]

    #forward pass and prediction
    ypred = nn.forward(x)
    print('True Class:     ', np.argmax(Y[i]))
    print('Predicted Class:', np.argmax(ypred),'\n')

    #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
    mask = np.zeros_like(ypred)
    mask[:,np.argmax(ypred)] = 1
    Rinit = ypred*mask


    #compute first layer relevance according to prediction
    #R = nn.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
    R = nn.lrp(Rinit,'epsilon',1.)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
    #R = nn.lrp(Rinit,'alphabeta',2)    #as Eq(60) from DOI: 10.1371/journal.pone.0130140

    #R = nn.lrp(ypred*Y[na,i],'epsilon',1.) #compute first layer relevance according to the true class label


    '''
    #compute first layer relvance for an arbitrarily selected class
    for yselect in range(10):
        yselect = (np.arange(Y.shape[1])[na,:] == yselect)*1.
        R = nn.lrp(ypred*yselect,'epsilon',0.1)
    '''

    '''
    # you may also specify different decompositions for each layer, e.g. as below:
    # first, set all layers (by calling set_lrp_parameters on the container module
    # of class Sequential) to perform alpha-beta decomposition with alpha = 1.
    # this causes the resulting relevance map to display excitation potential for the prediction
    #
    nn.set_lrp_parameters('alpha',1.)
    #
    # set the first layer (a convolutional layer) decomposition variant to 'w^2'. This may be especially
    # usefill if input values are ranged [0 V], with 0 being a frequent occurrence, but one still wishes to know about
    # the relevance feedback propagated to the pixels below the filter
    # the result with display relevance in important areas despite zero input activation energy.
    #
    nn.modules[0].set_lrp_parameters('ww') # also try 'flat'
    # compute the relevance map
    R = nn.lrp(Rinit)
    '''



    #sum over the third (color channel) axis. not necessary here, but for color images it would be.
    R = R.sum(axis=3)
    #same for input. create brightness image in [0,1].
    xs = ((x+1.)/2.).sum(axis=3)

    #render input and heatmap as rgb images
    digit = render.digit_to_rgb(xs, scaling = 3)
    hm = render.hm_to_rgb(R, X = xs, scaling = 3, sigma = 2)
    digit_hm = render.save_image([digit,hm],'../heatmap.png')
    data_io.write(R,'../heatmap.npy')

    #display the image as written to file
    plt.imshow(digit_hm, interpolation = 'none')
    plt.axis('off')
    plt.show()


#note that modules.Sequential allows for batch processing inputs
'''
x = X[:10,...]
y = nn.forward(x)
R = nn.lrp(y)
data_io.write(R,'../Rbatch.npy')
'''
