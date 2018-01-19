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

import numpy as np
from .module import Module

# -------------------------------
# Softmax layer
# -------------------------------
class SoftMax(Module):
    '''
    Softmax Layer
    '''

    def __init__(self):
        Module.__init__(self)

    def forward(self,X,*args,**kwargs):
        self.X = X
        self.Y = np.exp(X) / np.exp(X).sum(axis=1,keepdims=True)
        return self.Y


    def lrp(self,R,*args,**kwargs):
        # just propagate R further down.
        # makes sure subroutines never get called.
        #return R*self.X
        return R

    def clean(self):
        self.X = None
        self.Y = None