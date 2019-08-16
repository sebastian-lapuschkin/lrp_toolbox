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

from .module import Module
import numpy
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"):
    import cupy
    import cupy as np
na = np.newaxis

# -------------------------------
# Rectification layer
# -------------------------------
class Rect(Module):
    '''
    Rectification Layer
    '''
    def __init__(self):
        Module.__init__(self)

    def to_cupy(self):
        global np
        assert imp.find_spec("cupy"), "module cupy not found."
        if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.array(self.Y)
        np = cupy

    def to_numpy(self):
        global np
        if not imp.find_spec("cupy"):
            pass #nothing to do if there is no cupy. model should exist as numpy arrays
        else:
            if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.asnumpy(self.Y)
            np = numpy

    def forward(self,X,*args,**kwargs ):
        self.Y = np.maximum(0,X)
        return self.Y

    def backward(self,DY):
        return DY*(self.Y!=0)

    def clean(self):
        self.Y = None

    def lrp(self,R,*args,**kwargs):
        # component-wise operations within this layer
        # ->
        # just propagate R further down.
        # makes sure subroutines never get called.
        return R

