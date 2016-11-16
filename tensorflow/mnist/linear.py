import tensorflow as tf
from module import Module
import variables


class linear():
    '''
    Linear Layer
    '''

    def __init__(self, input_tensor, output_dim, activation_bool=False, activation_fn=tf.nn.relu,name="linear"):

        #Module.__init__(self)

        self.input_tensor = input_tensor
        self.input_shape = self.input_tensor.get_shape().as_list()
        self.output_dim = output_dim
        self.check_input_shape()

        self.weights_shape = [self.input_shape[-1], self.output_dim]
        with tf.variable_scope(name):
            self.weights = variables.weights(self.weights_shape)
            self.biases = variables.biases(self.output_dim)

    def __new__(self):
        with tf.name_scope('activations'):
            linear = tf.matmul(self.input_tensor, self.weights)
            activations = tf.nn.bias_add(linear, self.biases)
            #activations = activation_fn(conv, name='activation')
            tf.histogram_summary(name + '/activations', activations)
        return activations

    def check_input_shape(self):
        if len(self.input_shape)!=2:
            raise ValueError('Expected dimension of input tensor: 2')

    # def forward(self):
    #     with tf.name_scope('activations'):
    #         linear = tf.matmul(self.input_tensor, self.weights)
    #         activations = tf.nn.bias_add(linear, self.biases)
    #         #activations = activation_fn(conv, name='activation')
    #         tf.histogram_summary(name + '/activations', activations)

    #     return activations

    def lrp(self):
        return 0
        
