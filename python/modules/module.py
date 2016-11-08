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

    def __init__(self):
        ''' The constructor '''
        pass

    def update(self, lrate):
        ''' update should implement the layer parameter updating step '''
        pass

    def clean(self):
        ''' clean can be used to remove any temporary variables from the layer, e.g. just before serializing the layer object'''
        pass

    def lrp(self,R, *args, **kwargs):
        ''' entry point for the lrp backward pass '''
        return R

    def backward(self,DY):
        ''' backward passes the error gradient DY to the input neurons '''
        return DY

    def train(self, X, Y, *args, **kwargs):
        ''' implements (currently in modules.Sequential only) a simple training routine '''

    def forward(self,X):
        ''' forward passes the input data X to the layer's output neurons '''
        return X