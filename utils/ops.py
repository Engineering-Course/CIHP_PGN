import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

def conv2d(input_, output, kernel, stride, relu, bn, name, stddev=0.01):
    with tf.variable_scope(name) as scope:
    # Convolution for a given input and kernel
        shape = [kernel, kernel, input_.get_shape()[-1], output]
        w = tf.get_variable('w', shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME')
        # Add the biases
        b = tf.get_variable('b', [output], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        if bn:
            conv = tf.layers.batch_normalization(conv)
        # ReLU non-linearity
        if relu:
            conv = tf.nn.relu(conv, name=scope.name)
        return conv

def max_pool(input_, kernel, stride, name):
    return tf.nn.max_pool(input_, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

def linear(input_, output, name, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        matrix = tf.get_variable("Matrix", [shape[1], output], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

def atrous_conv2d(input_, output, kernel, rate, relu, name, stddev=0.01):
    with tf.variable_scope(name) as scope:
    # Dilation convolution for a given input and kernel
        shape = [kernel, kernel, input_.get_shape()[-1], output]
        w = tf.get_variable('w', shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.atrous_conv2d(input_, w, rate, padding='SAME')
        # Add the biases
        b = tf.get_variable('b', [output], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        # ReLU non-linearity
        if relu:
            conv = tf.nn.relu(conv, name=scope.name)
        return conv

def gcn(input_, output, kernel, stride, relu, bn, name, stddev=0.01):
    with tf.variable_scope(name) as scope:
        left_shape_k_1 = [kernel, 1, input_.get_shape()[-1], output]
        left_shape_1_k = [1, kernel, output, output]
        right_shape_1_k = [1, kernel, input_.get_shape()[-1], output]
        right_shape_k_1 = [kernel, 1, output, output]
        w1_1 = tf.get_variable('w1_1', left_shape_k_1, initializer=tf.truncated_normal_initializer(stddev=stddev))
        w1_2 = tf.get_variable('w1_2', left_shape_1_k, initializer=tf.truncated_normal_initializer(stddev=stddev))
        w2_1 = tf.get_variable('w2_1', right_shape_1_k, initializer=tf.truncated_normal_initializer(stddev=stddev))
        w2_2 = tf.get_variable('w2_2', right_shape_k_1, initializer=tf.truncated_normal_initializer(stddev=stddev))
        b1_1 = tf.get_variable('b1_1', [output], initializer=tf.constant_initializer(0.0))
        b1_2 = tf.get_variable('b1_2', [output], initializer=tf.constant_initializer(0.0))
        b2_1 = tf.get_variable('b2_1', [output], initializer=tf.constant_initializer(0.0))
        b2_2 = tf.get_variable('b2_2', [output], initializer=tf.constant_initializer(0.0))

        conv1_1 = tf.nn.conv2d(input_, w1_1, strides=[1, stride, stride, 1], padding='SAME')
        conv1_1 = tf.nn.bias_add(conv1_1, b1_1)
        if bn:
            conv1_1 = tf.layers.batch_normalization(conv1_1)
        if relu:
            conv1_1 = tf.nn.relu(conv1_1, name=scope.name)        

        conv1_2 = tf.nn.conv2d(conv1_1, w1_2, strides=[1, stride, stride, 1], padding='SAME')
        conv1_2 = tf.nn.bias_add(conv1_2, b1_2)
        if bn:
            conv1_2 = tf.layers.batch_normalization(conv1_2)
        if relu:
            conv1_2 = tf.nn.relu(conv1_2, name=scope.name)
        
        conv2_1 = tf.nn.conv2d(input_, w2_1, strides=[1, stride, stride, 1], padding='SAME')
        conv2_1 = tf.nn.bias_add(conv2_1, b2_1)
        if bn:
            conv2_1 = tf.layers.batch_normalization(conv2_1)
        if relu:
            conv2_1 = tf.nn.relu(conv2_1, name=scope.name)

        conv2_2 = tf.nn.conv2d(conv2_1, w2_2, strides=[1, stride, stride, 1], padding='SAME')
        conv2_2 = tf.nn.bias_add(conv2_2, b2_2)
        if bn:
            conv2_2 = tf.layers.batch_normalization(conv2_2)
        if relu:
            conv2_2 = tf.nn.relu(conv2_2, name=scope.name)

        top = tf.add_n([conv1_2, conv2_2])

        return top

def br(input_, output, kernel, stride, name):
    with tf.variable_scope(name) as scope:

        br_conv1 = conv2d(input_, output, kernel, stride, relu=True, bn=False, name='br_conv1')
        br_conv2 = conv2d(br_conv1, output, kernel, stride, relu=False, bn=False, name='br_conv2')
        top = tf.add_n([input_, br_conv2])

        return top

def residual_module(input_, output, is_BN, name):
    mid_channel = output >> 1
    with tf.variable_scope(name) as scope:
        conv1 = conv2d(input_, mid_channel, 1, 1, relu=True, bn=is_BN, name='res_conv1')
        conv2 = conv2d(conv1, mid_channel, 3, 1, relu=True, bn=is_BN, name='res_conv2')
        conv3 = conv2d(conv2, output, 1, 1, relu=False, bn=is_BN, name='res_conv3')
        conv_side = conv2d(input_, output, 1, 1, relu=False, bn=is_BN, name='res_conv_side')
        top = tf.add_n([conv3, conv_side])
        top = tf.nn.relu(top, name=scope.name)

    return top 

def gcn_residual_module(input_, output, gcn_kernel, is_BN, name):
    mid_channel = output >> 1
    with tf.variable_scope(name) as scope:
        gcn_layer = gcn(input_, mid_channel, gcn_kernel, 1, relu=True, bn=is_BN, name='gcn_residual1')
        conv1 = conv2d(gcn_layer, output, 1, 1, relu=False, bn=is_BN, name='gcn_residual2')
        conv_side = conv2d(input_, output, 1, 1, relu=False, bn=-is_BN, name='gcn_residual3')
        top = tf.add_n([conv1, conv_side])
        top = tf.nn.relu(top, name=scope.name)
    return top