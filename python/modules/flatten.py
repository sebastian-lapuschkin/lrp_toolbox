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

# -------------------------------
# Flattening Layer
# -------------------------------

class Flatten(Module):

    def __init__(self):
        self.inputshape = []

    def backward(self,DY):
        return np.reshape(DY,self.inputshape)

    def forward(self,X):
        self.inputshape = X.shape # N x H x W x D
        return np.reshape(X,[self.inputshape[0],np.prod(self.inputshape[1:])])


    def lrp(self,R, *args, **kwargs):
        return np.reshape(R,self.inputshape)