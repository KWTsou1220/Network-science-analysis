from layer import linear_layer, pooling1d_layer
from layer import conv_layer, conv1d_layer
import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, LR, input_shape, output_shape, model_name='Model', optimizer=tf.train.AdamOptimizer):
        
        # optimization setting
        self.LR = LR
        self.optimizer = optimizer
        
        # naming setting
        self.model_name = model_name
        
        # model setting
        self.input_shape = input_shape
        self.output_shape = output_shape
        with tf.variable_scope(self.model_name):
            self.x = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t = tf.placeholder(dtype=tf.float32, shape=self.output_shape, name='target')
            self.drop_prob = tf.placeholder(tf.float32)
            self.y = self._forward_pass(self.x)
            
    
    def optimize(self, loss, *args):
        if len(args) > 0:
            self.loss = loss(self.y, self.t, args)
        else:
            self.loss = loss(self.y, self.t)
        self.train_model = self.optimizer(self.LR).minimize(self.loss) 
        
class DNN(Model):
    def _forward_pass(self, x):
        h1 = linear_layer(x, shape=200, name='L1')
        h1 = tf.nn.dropout(h1, 1-self.drop_prob)
        h2 = linear_layer(h1, shape=100, name='L2')
        h2 = tf.nn.dropout(h2, 1-self.drop_prob)
        h3 = linear_layer(h2, shape=50, name='L3')
        h3 = tf.nn.dropout(h3, 1-self.drop_prob)
        h4 = linear_layer(h3, shape=self.output_shape[1], name='L4')
        return h4
        
class CNN1d(Model):
    
    def _forward_pass(self, x):
        # Convolutional Layer
        h1 = conv1d_layer(x, filter_shape=[3, 11, 32], name='L1') # 1500
        h2 = conv1d_layer(h1, filter_shape=[3, 32, 32], name='L2')
        h3 = pooling1d_layer(h2, pool_size=[2], strides=[2], name='L3') # 750
        h4 = conv1d_layer(h3, filter_shape=[3, 32, 64], name='L4')
        h5 = conv1d_layer(h4, filter_shape=[3, 64, 64], name='L5')
        h6 = pooling1d_layer(h5, pool_size=[2], strides=[2], name='L6') # 325
        h7 = conv1d_layer(h6, filter_shape=[3, 64, 128], name='L7')
        h8 = conv1d_layer(h7, filter_shape=[3, 128, 128], name='L8')
        h9 = pooling1d_layer(h8, pool_size=[2], strides=[2], name='L9') # 163
        h10 = conv1d_layer(h9, filter_shape=[3, 128, 256], name='L10')
        h11 = conv1d_layer(h10, filter_shape=[3, 256, 256], name='L11')
        
        # Fully-connected layer
        _, width, ch = h11.get_shape().as_list()
        h11 = tf.reshape(h11, [-1, width*ch])
        h12 = linear_layer(h11, shape=1000, name='L12')
        h13 = linear_layer(h12, shape=100, name='L13')
        h14 = linear_layer(h13, shape=5, name='L14', non_linear=None)
        return h14
    
class CNN(Model):
    
    def _forward_pass(self, x):
        # Convolutional Layer
        h1 = conv_layer(x, filter_shape=[3, 3, 11, 32], name='L1') # (100, 29)
        h2 = conv_layer(h1, filter_shape=[3, 3, 32, 32], strides=[1, 2, 1, 1], name='L2') # (50, 29)
        h3 = conv_layer(h2, filter_shape=[3, 3, 32, 64], name='L3')
        h4 = conv_layer(h3, filter_shape=[3, 3, 64, 64], strides=[1, 2, 1, 1], name='L4') # (25, 29)
        h5 = conv_layer(h4, filter_shape=[3, 3, 64, 128], name='L5')
        h6 = conv_layer(h5, filter_shape=[3, 3, 128, 128], strides=[1, 2, 1, 1], name='L6') # (13, 29)
        h7 = conv_layer(h6, filter_shape=[3, 3, 128, 256], name='L7')
        h8 = conv_layer(h7, filter_shape=[3, 3, 256, 256], strides=[1, 1, 2, 1], name='L8') # (13, 15)
        
        # Fully-connected layer
        _, height, width, ch = h8.get_shape().as_list()
        h8 = tf.reshape(h8, [-1, height*width*ch])
        h9 = linear_layer(h8, shape=1000, name='L9')
        h10 = linear_layer(h9, shape=100, name='L10')
        h11 = linear_layer(h10, shape=5, name='L14', non_linear=None)
        return h11
    
    
    