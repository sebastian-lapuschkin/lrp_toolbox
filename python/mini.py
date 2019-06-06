'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 21.09.2015
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''


# imports
import model_io
import data_io
import render

import importlib.util as imp
import numpy
import numpy as np
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
na = np.newaxis
# end of imports

nn = model_io.read('../models/MNIST/long-rect.nn') # read model
X = data_io.read('../data/MNIST/test_images.npy')[na,0,:] # load first MNIST test image
X = X / 127.5 - 1 # normalized data to range [-1 1]

Ypred = nn.forward(X) # forward pass through network
R = nn.lrp(Ypred) # lrp to explain prediction of X

if not np == numpy: # np=cupy
        X = np.asnumpy(X)
        R = np.asnumpy(R)

# render rgb images and save as image
digit = render.digit_to_rgb(X)
hm = render.hm_to_rgb(R, X) # render heatmap R, use X as outline
render.save_image([digit, hm], '../hm_py.png')
