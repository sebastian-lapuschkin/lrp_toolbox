import tensorflow as tf
from module import Module



class relu(Module):
    '''
    Relu Layer
    '''

    def __init__(self):
        Module.__init__(self)

    def forward(self,input_tensor):
        with tf.variable_scope(name):
            with tf.name_scope('activations'):
                self.activations = tf.nn.relu(input_tensor, name='relu')
            tf.histogram_summary(name + '/activations', self.activations)
        return self.activations

    def clean(self):
        self.activations = None

    def lrp(self,R,*args,**kwargs):
        # component-wise operations within this layer
        # ->
        # just propagate R further down.
        # makes sure subroutines never get called.
        return R
