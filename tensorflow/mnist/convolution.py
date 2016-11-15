import tensorflow as tf
from module import Module



class convolution(Module):
    '''
    Convolutional Layer
    '''

    def __init__(self, input_tensor, output_dim, kernel_height=5, kernel_width=5, stride_height=2, stride_width=2, activation_bool=False, activation_fn=tf.nn.relu, pad = 'SAME',name="conv2d"):

        Module.__init__(self)

        self.input_tensor = input_tensor
        self.input_shape = self.input_tensor.get_shape().as_list()
        self.output_dim = output_dim
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.strides = [1,stride_height, stride_width,1]
        self.check_shape()

        wb = Weights()
        self.weights, self.biases = wb.weights, wb.biases

    def check_shape(self):
        if len(self.input_shape)!=4:
            raise ValueError('Expected dimension of input tensor: 4')

    def forward(self):
        with tf.name_scope('activations'):
            conv = tf.nn.conv2d(self.input_tensor, self.weights, strides = self.strides, padding=pad)
            activations = tf.reshape(tf.nn.bias_add(conv, self.biases), conv.get_shape())
            #activations = activation_fn(conv, name='activation')
            tf.histogram_summary(name + '/activations', activations)

        return activations

    def lrp(self):
        return 0
        
