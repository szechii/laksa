'''
author:         szechi
date created:   2 Oct 2018
description:    common nn blocks for model building
'''
import os
import sys
import numpy as np
import tensorflow as tf

def conv3d(input_T, output_channels, kernel_dim, stride, padding='SAME', scope=None):
    if scope is None:
        scope = "conv3d"
    with tf.name_scope(scope):
        input_channels = int(input_T.shape[-1]) # Last dim is # channels
        filter_dim = kernel_dim + (input_channels, output_channels)
        bias = tf.Variable(tf.zeros([output_channels]))
        filter_F = tf.Variable(tf.truncated_normal(filter_dim, stddev=0.1, seed=1234))
        conv_output = tf.nn.conv3d(input_T, filter_F, \
            strides=(1,)+stride+(1,), padding=padding)
        final_output = tf.nn.bias_add(conv_output, bias)
        return final_output

def atrousconv3d(input_T, output_channels, kernel_dim, stride, dilation_rate, padding='SAME', scope=None):
    if scope is None:
        scope = "atrousconv3d"
    with tf.name_scope(scope):
        input_channels = int(input_T.shape[-1]) # Last dim is # channels
        filter_dim = kernel_dim + (input_channels, output_channels)
        tf_drate = (dilation_rate, dilation_rate)
        bias = tf.Variable(tf.zeros([output_channels]))
        filter_F = tf.Variable(tf.truncated_normal(filter_dim, stddev=0.1, seed=1234))
        conv_output = tf.nn.convolution(input_T, filter_F, \
            padding=padding, strides=stride, dilation_rate=tf_drate)
        final_output = tf.nn.bias_add(conv_output, bias)
        return final_output

# fn to initialize transposed convolution kernel as bilinear interpolation
# Assume all in_channels are mapped to out_channels via bilinear interpolation
def bilinear_2d_kernel(shape):
    # Set up 2D bilinear kernel
    def upkernel_bilinear(upkernel_2D_dim):
        # Get center of kernel taking into account odd/even centers
        fac = ((np.array(upkernel_2D_dim)+1)/2).astype(int)
        ctr = (np.array(upkernel_2D_dim)+1.0)/2
        ogrid = np.ogrid[1-ctr[0]:ctr[0], 1-ctr[1]:ctr[1]]
        # Generate pyramid with peak at kernel center
        ogrid = (1-abs(1.0*ogrid[0])/fac[0])*(1-abs(ogrid[1])/fac[1])
        return ogrid

    # kernel = [h,w,out_channels,in_channels]
    W_kernel = upkernel_bilinear(shape[1:3])
    W_out = np.zeros(shape)
    for i_out in xrange(0,shape[3]):
        for i_in in xrange(0,shape[4]):
            W_out[:,:,:,i_out,i_in] = W_kernel
    return W_out.astype(np.float32)

def deconv3d(input_T, output_channels, kernel_dim, stride, depth_value, padding='SAME', scope=None):
    if scope is None:
        scope = "deconv3d"
    with tf.name_scope(scope):
        output_shape = (tf.shape(input_T)[0], depth_value, \
            2*int(input_T.shape[2].value), 2*int(input_T.shape[3].value), \
            output_channels)
        input_channels = int(input_T.shape[-1].value)
        filter_dim = kernel_dim + (output_channels, input_channels)

        bias = tf.Variable(tf.zeros([output_channels]))
        filter_F = tf.Variable(bilinear_2d_kernel(filter_dim), tf.float32)

        deconv_output = tf.nn.conv3d_transpose(input_T, filter_F, \
            output_shape, strides=(1,)+stride+(1,), padding=padding)
        final_output = tf.nn.bias_add(deconv_output, bias)
        return final_output

def maxpool(input, kernel_dim, stride, padding='SAME', scope=None):
    if scope is None:
        scope = "maxpool"
    with tf.name_scope(scope):
        output = tf.nn.max_pool3d(input, (1,)+kernel_dim+(1,), \
            (1,)+stride+(1,), padding=padding)
        return output

def avgpool(input, kernel_dim, stride, padding='SAME', scope=None):
    if scope is None:
        scope = "avgpool"
    with tf.name_scope(scope):
        output = tf.nn.avg_pool3d(input, (1,)+kernel_dim+(1,), \
            (1,)+stride+(1,), padding=padding)
        return output

def dropout(y, training, pkeep, scope=None):
    if scope is None:
        scope = "dropout"
    with tf.name_scope(scope):
        dp = tf.nn.dropout(y, pkeep)
        return dp

def batchnorm(input_layer, training, scope=None):
    if scope is None:
        scope = "batchnorm"
    var_scope = os.path.join(tf.get_default_graph().get_name_scope(), scope)
    bn = tf.layers.batch_normalization(input_layer, axis=-1, \
        training=training, name=var_scope)
    return bn

def lrn3d(input, scope=None):
    if scope is None:
        scope = "lrn3d"
    with tf.name_scope(scope):
        lrn = tf.nn.lrn(input)
        return lrn

def convBNReLUDrop(input_layer, filtsize, ksize, stride, training, kprob, dilation_rate, scope=None):
    if scope is None:
        scope = "convBNReLUDrop"
    with tf.name_scope(scope):
        conv01 = conv3d(input_layer, filtsize, ksize, stride, padding='SAME')
        # conv01 = atrousconv3d(input_layer, filtsize, ksize, stride, dilation_rate, padding='SAME')
        conv01_BN = batchnorm(conv01, training)
        conv01_BN_RELU = tf.nn.relu(conv01_BN)
        conv01_BN_RELU_DROP = dropout(conv01_BN_RELU, training, kprob)
        return conv01_BN_RELU_DROP

def resblock(input_layer, filtsize, ksize, stride, training, kprob, dilation_rate, scope=None):
    if scope is None:
        scope = "resblock"
    var_scope = os.path.join(tf.get_default_graph().get_name_scope(), scope)
    with tf.name_scope(var_scope):
        input_channels = input_layer.shape[-1].value
        conv01 = convBNReLUDrop(input_layer, filtsize, ksize, stride, training, kprob, dilation_rate, scope='s01')
        conv02 = convBNReLUDrop(conv01, filtsize, ksize, stride, training, kprob, dilation_rate, scope='s02')
        if input_channels != filtsize:
            rechannel = conv3d(input_layer, filtsize, (1,1,1), stride, padding='SAME')
        else:
            rechannel = input_layer
        skip03 = tf.add(conv02, rechannel)
        conv03 = convBNReLUDrop(skip03, filtsize, ksize, stride, training, kprob, dilation_rate, scope='s03')
        return conv03

def fcn(input_layer, num_outputs, training, kprob, scope=None, reuse=False, \
    activation=tf.nn.relu):
    if scope is None:
        scope = "fcnblock"
    var_scope = os.path.join(tf.get_default_graph().get_name_scope(), scope)
    fcn_in = tf.contrib.layers.flatten(input_layer)
    num_inputs = fcn_in.shape[-1].value

    fcn_out = tf.layers.dense(fcn_in, num_outputs, activation=activation,
        kernel_initializer=tf.truncated_normal_initializer(), \
        bias_initializer=tf.truncated_normal_initializer(), \
        kernel_regularizer=None, \
        bias_regularizer=None, \
        activity_regularizer=None, kernel_constraint=None, use_bias=True, \
        bias_constraint=None, trainable=True, name=var_scope, reuse=reuse)
    fcn_out = dropout(fcn_out, training, kprob)
    return fcn_out
