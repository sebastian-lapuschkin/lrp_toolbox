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
from module import Module

# -------------------------------
# Tanh layer
# -------------------------------
class Tanh(Module):
    '''
    Tanh Layer
    '''

    def __init__(self):
        Module.__init__(self)

    def forward(self,X):
        self.Y = np.tanh(X)
        return self.Y


    def backward(self,DY):
        return DY*(1.0-self.Y**2)


    def clean(self):
        self.Y = None

    def lrp(self,R,*args,**kwargs):
        # component-wise operations within this layer
        # ->
        # just propagate R further down.
        # makes sure subroutines never get called.
        return R