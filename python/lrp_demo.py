'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

The purpose of this module is to demonstrate the process of obtaining pixel-wise explanations for given data points at hand of the MNIST hand written digit data set.

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
nn = model_io.read('../models/MNIST/long-rect.nn') # 99.17% prediction accuracy
X = data_io.read('../data/MNIST/test_images.npy')
Y = data_io.read('../data/MNIST/test_labels.npy')

# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
X =  X / 127.5 - 1

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1

#permute data order for demonstration. or not. your choice.
I = np.arange(X.shape[0])
#I = np.random.permutation(I)

#predict and perform LRP for the 10 first samples
for i in I[:10]:
    x = X[na,i,:]

    #forward pass and prediction
    ypred = nn.forward(x)
    print('True Class:     ', np.argmax(Y[i]))
    print('Predicted Class:', np.argmax(ypred),'\n')


    #compute first layer relevance according to prediction
    #R = nn.lrp(ypred)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
    R = nn.lrp(ypred,'epsilon',1.)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
    #R = nn.lrp(ypred,'alphabeta',2)    #as Eq(60) from DOI: 10.1371/journal.pone.0130140


    #R = nn.lrp(Y[na,i]) #compute first layer relevance according to the true class label


    '''
    yselect = 3
    yselect = (np.arange(Y.shape[1])[na,:] == yselect)*1.
    R = nn.lrp(yselect) #compute first layer relvance for an arbitrarily selected class
    '''

    #undo input normalization for digit drawing. get it back to range [0,1] per pixel
    x = (x+1.)/2.

    #render input and heatmap as rgb images
    digit = render.digit_to_rgb(x, scaling = 3)
    hm = render.hm_to_rgb(R, X = x, scaling = 3, sigma = 2)
    digit_hm = render.save_image([digit,hm],'../heatmap.png')
    data_io.write(R,'../heatmap.npy')

    #display the image as written to file
    plt.imshow(digit_hm, interpolation = 'none')
    plt.axis('off')
    plt.show()


#note that modules.Sequential allows for batch processing inputs
'''
x = X[:10,:]
y = nn.forward(x)
R = nn.lrp(y)
data_io.write(R,'../Rbatch.npy')
'''
