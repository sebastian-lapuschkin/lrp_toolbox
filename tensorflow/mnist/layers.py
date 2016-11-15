import tensorflow as tf

import pdb

def weight_variable(weights_shape, initializer=tf.truncated_normal_initializer(stddev=0.01), name = 'weights'):
    """Define a weight variable """
    return tf.get_variable(name, shape=weights_shape, initializer=initializer)

def bias_variable(biases_shape, initializer = 0, name = 'biases'):
    """Define a bias variable """
    return tf.get_variable(name, biases_shape, initializer=tf.constant_initializer(initializer))

'''Linear Functions '''
def linear(input_tensor, output_dim, dropout_bool=False, dropout_prob=0.8, activation_bool=False, activation_fn=tf.nn.relu, name='layer'):
    """Fully Connected Layer - Performs y = xw + b"""
    #check for  correct shape of input tensor
    input_shape = input_tensor.get_shape().as_list()
    if len(input_shape)==2:
        input_dim = input_shape[1]
    else:
        raise ValueError('Expected dimension of input tensor: 2')

    #get all variables under scope of layer name
    with tf.variable_scope(name):
        #define variables
        weights_shape = [input_dim, output_dim]
        weights = weight_variable(weights_shape, name='weights')
        biases = bias_variable([output_dim], name= 'biases')

        #get output activations
        with tf.name_scope('activations'):
            activations = activation_fn((tf.matmul(input_tensor, weights) + biases), name='activation')
            tf.histogram_summary(name + '/activations', activations)

    #apply dropout if training
    if dropout_bool:
        with tf.name_scope('dropout'):
            tf.scalar_summary('dropout_keep_probability', dropout_prob)
            activations = tf.nn.dropout(activations, dropout_prob)
    return activations

'''Convolution Functions '''
def conv2d(input_tensor, output_dim, kernel_height=5, kernel_width=5, stride_height=2, stride_width=2, activation_bool=False, activation_fn=tf.nn.relu, pad = 'SAME',name="conv2d"):

    #check for  correct shape of input tensor
    input_shape = input_tensor.get_shape().as_list()
    if len(input_shape)!=4:
        raise ValueError('Expected dimension of input tensor: 4')

    #get all variables under scope of layer name
    with tf.variable_scope(name):
        #define variables
        #pdb.set_trace()
        weights_shape = [kernel_height, kernel_width, input_shape[-1], output_dim]
        weights = weight_variable(weights_shape, name='weights')
        biases = bias_variable([output_dim], name= 'biases')

        #get output activations
        with tf.name_scope('activations'):
            conv = tf.nn.conv2d(input_tensor, weights, strides=[1,stride_height, stride_width,1], padding=pad)
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            activations = activation_fn(conv, name='activation')
            tf.histogram_summary(name + '/activations', activations)

    return activations         

def deconv2d(input_tensor, output_dim, kernel_height=5, kernel_width=5, stride_height=2, stride_width=2, activation_bool=False, activation_fn=tf.nn.relu, name="conv2d"):

    #check for  correct shape of input tensor
    input_shape = input_tensor.get_shape().as_list()
    if len(input_shape)!=4:
        raise ValueError('Expected dimension of input tensor: 4')
    if len(output_dim)!=4:
        raise ValueError('Expected dimension of output dimension: 4')

    #get all variables under scope of layer name
    with tf.variable_scope(name):
        #define variables
        #pdb.set_trace()
        weights_shape = [kernel_height, kernel_width, output_dim[-1], input_shape[-1]]
        weights = weight_variable(weights_shape, name='weights')
        biases = bias_variable([output_dim[-1]], name= 'biases')

        #get output activations
        with tf.name_scope('activations'):
            deconv = tf.nn.deconv2d(input_tensor, weights, output_shape=output_dim, strides=[1,stride_height, stride_width,1])
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
            activations = activation_fn(deconv, name='activation')
            tf.histogram_summary(name + '/activations', activations)

    return activations    


''' Activation Functions'''
def activation(input_tensor, activation_fn, name='activations'):
    return activation_fn(input_tensor)
    
def tanh(input_tensor, name='tanh'):
    with tf.variable_scope(name):
        with tf.name_scope('activations'):
            activations = activation(input_tensor, tf.nn.tanh, name='tanh')
            tf.histogram_summary(name + '/activations', activations)
    return activations

def relu(input_tensor, name='relu'):
    with tf.variable_scope(name):
        with tf.name_scope('activations'):
            activations = activation(input_tensor, tf.nn.relu, name='relu')
            tf.histogram_summary(name + '/activations', activations)
    return activations

def softmax(input_tensor, name='softmax'):
    with tf.variable_scope(name):
        with tf.name_scope('activations'):
            activations = activation(input_tensor, tf.nn.softmax, name='softmax')
            tf.histogram_summary(name + '/activations', activations)
    return activations

''' Pooling Functions'''
def pool(input_tensor, pool_fn, kernel, strides, padding, name='pool' ):
    #check for  correct shape of input tensor
    kernel_shape = kernel.get_shape().as_list()
    strides_shape = strides.get_shape().as_list()
    if len(kernel_shape)!=4:
        raise ValueError('Expected dimension of input tensor: 4')
    if len(strides_shape)!=4:
        raise ValueError('Expected dimension of output dimension: 4')
    return pool_fn(input_tensor, kernel, strides, padding=padding, name=name)

def max_pool(input_tensor, kernel=[1,2,2,1], strides=[1,2,2,1], pad='SAME', name='max_pool'):
    #get all variables under scope of layer name
    with tf.variable_scope(name):
        with tf.name_scope('activations'):
            activations = pool(input_tensor, tf.nn.max_pool, kernel, strides, padding=pad, name='max_pool')
            tf.histogram_summary(name + '/activations', activations)
    return activations

def average_pool(input_tensor, kernel=[1,2,2,1], strides=[1,2,2,1], pad='SAME', name='average_pool'):
    
    #get all variables under scope of layer name
    with tf.variable_scope(name):
        with tf.name_scope('activations'):
            activations = pool(input_tensor, tf.nn.max_pool, kernel, strides, padding=pad, name='average_pool')
            tf.histogram_summary(name + '/activations', activations)
    return activations














            
