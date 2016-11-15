import tensorflow as tf



class weights():
    def __init__(self, weights_shape, initializer=tf.truncated_normal_initializer(stddev=0.01), name='weights'):
        self.weights_shape = weights_shape
        self.initial = initializer
        self.name = name
    def __new__(self):
        return tf.get_variable(self.name, shape=self.weights_shape, initializer=self.initializer)

class biases():
    def __init__(self, bias_shape, initializer = 0, name = 'biases'):
        self.bias_shape = bias_shape
        self.initializer = tf.constant_initializer(initializer)
        self.name = name
    def __new__(self):
        return tf.get_variable(self.name, self.bias_shape, initializer=self.initializer)

