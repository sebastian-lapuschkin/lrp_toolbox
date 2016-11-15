# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

#from tensorflow.examples.tutorials.mnist
import input_data

import sys
sys.path.append("../../python")
import render
import numpy as np
from layers import linear
import pdb

FLAGS = None

#def simple_lrp(R, X, W, b):
def simple_lrp(R, Z, b):
    #Z = tf.expand_dims(W, 0) * tf.expand_dims(X, -1)
    #pdb.set_trace()
    
    Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(tf.expand_dims(b, 0), 0)
    
    return tf.reduce_sum((Z / Zs) * tf.expand_dims(R, 1),2)

 
# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summariesvariable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('stddev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, relevance=True, dropout=False, keep_prob=0.8):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        #variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        #variable_summaries(biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      
    if dropout:
      with tf.name_scope('dropout'):
        
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        activations = tf.nn.dropout(activations, keep_prob)
    return activations


def forward(x):
    # hidden1 = nn_layer(x, 784, 500, 'layer1')
    # y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)
    hidden1 = linear(x, 500, name='layer1')
    y = linear(hidden1, 10, activation_fn = tf.identity, name='layer2')
  
    return [y, hidden1]

def lrp(activations):
    print('Relevancing .. ')
    #pdb.set_trace()
    noLayers = len(activations) - 1
    Relevances = tf.nn.softmax(activations[0]) * activations[0]
    for i in range(noLayers):
        for v in tf.trainable_variables():
            if v.name == 'model/layer'+str(noLayers-i)+'/weights:0':
                weights = v.value()
            if v.name == 'model/layer'+str(noLayers-i)+'/biases:0':
                biases = v.value()
        Z = tf.expand_dims(weights, 0) * tf.expand_dims(activations[i+1], -1)
        Relevances = simple_lrp(Relevances, Z, biases)
    return Relevances

def visualize(relevances, images_tensor):
    #pdb.set_trace()
    n, dim = relevances.shape
    heatmap = relevances.reshape([n,28,28,1])
    input_images = images_tensor.reshape([n,28,28,1])
    heatmaps = []
    for h,heat in enumerate(heatmap):
        input_image = input_images[h]
        maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
        #import matplotlib.pyplot as plt;
        #plt.imsave('relevances/relevance'+str(h)+'.jpg',maps)
        heatmaps.append(maps)
    R = np.array(heatmaps)
    with tf.name_scope('input_reshape'):
        img = tf.image_summary('input', tf.cast(R, tf.float32), n)
    return img.eval()

def train():
    # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

  with tf.Session() as sess:

    # Create a multilayer model.
  
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
    
    with tf.variable_scope('model'):
        activations = forward(x)
        RELEVANCE = lrp(activations+[x])
        y = activations[0] 

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()


    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(FLAGS.batch_size, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.next_batch(FLAGS.test_batch_size)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # test-set accuracy
            test_inp = feed_dict(False)
            summary, acc , relevance_test= sess.run([merged, accuracy, RELEVANCE], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
        else:  
            inp = feed_dict(True)
            summary, _ , relevance_= sess.run([merged, train_step, RELEVANCE], feed_dict=inp)
            train_writer.add_summary(summary, i)

    test_img_summary = visualize(relevance_test, test_inp[test_inp.keys()[0]])
    test_writer.add_summary(test_img_summary)
    test_writer.flush()

    train_img_summary = visualize(relevance_, inp[inp.keys()[0]])
    train_writer.add_summary(train_img_summary)
    train_writer.flush()

    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=500,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='Number of steps to run trainer.')
    parser.add_argument('--test_batch_size', type=int, default=100,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
    parser.add_argument('--summaries_dir', type=str, default='/tmp/mnist_logs',
                      help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()
