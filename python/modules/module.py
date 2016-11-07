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