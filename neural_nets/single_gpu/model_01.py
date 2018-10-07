'''
author:         szechi
date created:   2 Oct 2018
description:    model scripts for image segmentation and classification (3D)
'''

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('../)
from nnblocks import *

def image_segmentation_model(x_in, output_shape, training, kprob):
    '''
    Image segmentation model, modified U-net with dense blocks.

    Input Args:     x_in            - input for training ([N, D, H, W, C] float tensor)
                    output_shape    - output shape of the segmentation model e.g. [D, H, W, nclass] (np.array)
                    training        - training boolean (boolean tensor)
                    kprob           - keep probability for dropout (float tensor)

    Output Args:    output_seg      - output of the model ([N, D, H, W, nclass] tensor)

    '''
    print "MODEL image_segmentation_model"
    # Static  variables
    ksize = (3,3,3)
    stride = (1,1,1)
    filters = [8,16,32,64,128,256]
    depth = [88, 44, 22, 11]
    pool_window = (2,2,2)
    pool_stride = (2,2,2)
    nseg_class = output_shape[-1]

    with tf.name_scope('ImageSegmentationModel'):
        with tf.name_scope('block01'):
            blk01a = convBNReLUDrop(x_in, filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk01b = convBNReLUDrop(tf.concat([x_in, blk01a], axis=-1), filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk01c = convBNReLUDrop(tf.concat([x_in, blk01a, blk01b], axis=-1), filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk01 = convBNReLUDrop(blk01c, filters[2], ksize, stride, training, kprob, 1, scope='s01')

        with tf.name_scope('block02'):
            blk02 = maxpool(blk01, pool_window, pool_stride, padding='SAME', scope='s01')

        with tf.name_scope('block03'):
            blk03a = convBNReLUDrop(blk02, filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk03b = convBNReLUDrop(tf.concat([blk02, blk03a], axis=-1), filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk03c = convBNReLUDrop(tf.concat([blk02, blk03a, blk03b], axis=-1), filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk03 = convBNReLUDrop(blk03c, filters[3], ksize, stride, training, kprob, 1, scope='s01')

        with tf.name_scope('block04'):
            blk04 = maxpool(blk03, pool_window, pool_stride, padding='SAME', scope='s01')

        with tf.name_scope('block05'):
            blk05a = convBNReLUDrop(blk04, filters[4], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk05b = convBNReLUDrop(tf.concat([blk04, blk05a], axis=-1), filters[4], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk05c = convBNReLUDrop(tf.concat([blk04, blk05a, blk05b], axis=-1), filters[4], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk05 = convBNReLUDrop(blk05c, filters[4], ksize, stride, training, kprob, 1, scope='s01')

        with tf.name_scope('block10'):
            blk10a =  deconv3d(blk05, filters[3], ksize, pool_stride, depth[1], padding='SAME', scope='s01')
            blk10 = tf.concat([blk03,blk10a], axis=-1, name='s01')

        with tf.name_scope('block11'):
            blk11a = convBNReLUDrop(blk10, filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk11b = convBNReLUDrop(tf.concat([blk10, blk11a], axis=-1), filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk11c = convBNReLUDrop(tf.concat([blk10, blk11a, blk11b], axis=-1), filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk11 = convBNReLUDrop(blk11c, filters[3], ksize, stride, training, kprob, 1, scope='s01')

        with tf.name_scope('block12'):
            blk12a =  deconv3d(blk11, filters[2], ksize, pool_stride, depth[0], padding='SAME', scope='s01')
            blk12 = tf.concat([blk01,blk12a], axis=-1, name='s01')

        with tf.name_scope('block13'):
            blk13a = convBNReLUDrop(blk12, filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk13b = convBNReLUDrop(tf.concat([blk12, blk13a], axis=-1), filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk13c = convBNReLUDrop(tf.concat([blk12, blk13a, blk13b], axis=-1), filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk13 = convBNReLUDrop(blk13c, filters[2], ksize, stride, training, kprob, 1, scope='s01')

        with tf.name_scope('block14'):
            blk14a = convBNReLUDrop(blk13, filters[1], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk14b = convBNReLUDrop(tf.concat([blk13, blk14a], axis=-1), filters[1], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk14c = convBNReLUDrop(tf.concat([blk13, blk14a, blk14b], axis=-1), filters[1], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk14 = convBNReLUDrop(blk14c, filters[1], ksize, stride, training, kprob, 1, scope='s01')
            output_seg = conv3d(blk14, nseg_class, (1,1,1), stride, padding='SAME', scope='s03')

    return output_seg

def image_classification_model(x_in, output_shape, training, kprob):
    '''
    Image classification model, modified dense-net.

    Input Args:     x_in            - input for training ([N, D, H, W, C] float tensor)
                    output_shape    - output shape of the segmentation model e.g. [D, H, W, nclass] or [nclass] (np.array)
                    training        - training boolean (boolean tensor)
                    kprob           - keep probability for dropout (float tensor)

    Output Args:    output_seg      - output of the model ([nclass] tensor)

    '''
    print "MODEL image_classification_model"
    # Static  variables
    ksize1 = (3,3,3)
    ksize = (1,1,1)
    stride = (1,1,1)
    filters = [16,32,64,128,256,512,1024]
    depth = [40, 20, 10, 5]
    pool_window = (2,2,2)
    pool_stride = (2,2,2)
    nseg_class = output_shape[-1]

    with tf.name_scope('ImageClassificationModel'):

        with tf.name_scope('block00'):
            blk00 = convBNReLU(x_in, filters[1] , ksize1, (1,1,1), training, kprob, 1, scope='conv01')

        with tf.name_scope('block01'):
            blk01a = convBNReLUDrop(blk00, filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk01b = convBNReLUDrop(tf.concat([blk00, blk01a], axis=-1), filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk01c = convBNReLUDrop(tf.concat([blk00, blk01a, blk01b], axis=-1), filters[2], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk01d = convBNReLUDrop(tf.concat([blk00, blk01a, blk01b, blk01c], axis=-1), filters[2], ksize, stride, training, kprob, 1, scope='dense04')
            blk01 = convBNReLUDrop(blk01d, filters[2], ksize1, stride, training, kprob, 1, scope='dense05')

        with tf.name_scope('block02'):
            blk02 = avgpool(blk01, pool_window, pool_stride, padding='SAME', scope='s01') #20x20x20x64

        with tf.name_scope('block03'):
            blk03a = convBNReLUDrop(blk02, filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk03b = convBNReLUDrop(tf.concat([blk02, blk03a], axis=-1), filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk03c = convBNReLUDrop(tf.concat([blk02, blk03a, blk03b], axis=-1), filters[3], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk03d = convBNReLUDrop(tf.concat([blk02, blk03a, blk03b, blk03c], axis=-1), filters[3], ksize, stride, training, kprob, 1, scope='dense04')
            blk03 = convBNReLUDrop(blk03d, filters[3], ksize1, stride, training, kprob, 1, scope='dense05')

        with tf.name_scope('block04'):
            blk04 = avgpool(blk03, pool_window, pool_stride, padding='SAME', scope='s01') #10x10x10x128

        with tf.name_scope('block05'):
            blk05a = convBNReLUDrop(blk04, filters[4], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk05b = convBNReLUDrop(tf.concat([blk04, blk05a], axis=-1), filters[4], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk05c = convBNReLUDrop(tf.concat([blk04, blk05a, blk05b], axis=-1), filters[4], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk05d = convBNReLUDrop(tf.concat([blk04, blk05a, blk05b, blk05c], axis=-1), filters[4], ksize, stride, training, kprob, 1, scope='dense04')
            blk05 = convBNReLUDrop(blk05d, filters[4], ksize1, stride, training, kprob, 1, scope='dense05')

        with tf.name_scope('block06'):
            blk06 = avgpool(blk05, pool_window, pool_stride, padding='SAME', scope='s01') #5x5x5x256

        with tf.name_scope('block07'):
            blk07a = convBNReLUDrop(blk06, filters[5], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk07b = convBNReLUDrop(tf.concat([blk06, blk07a], axis=-1), filters[5], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk07c = convBNReLUDrop(tf.concat([blk06, blk07a, blk07b], axis=-1), filters[5], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk07d = convBNReLUDrop(tf.concat([blk06, blk07a, blk07b, blk07c], axis=-1), filters[5], ksize, stride, training, kprob, 1, scope='dense04')
            blk07 = convBNReLUDrop(blk07d, filters[5], ksize1, stride, training, kprob, 1, scope='dense05')

        with tf.name_scope('block08'):
            blk08 = avgpool(blk07, pool_window, pool_stride, padding='SAME', scope='s01') #2x2x2x512

        with tf.name_scope('block09'):
            blk09a = convBNReLUDrop(blk08, filters[6], ksize, (1,1,1), training, kprob, 1, scope='dense01')
            blk09b = convBNReLUDrop(tf.concat([blk08, blk09a], axis=-1), filters[6], ksize, (1,1,1), training, kprob, 1, scope='dense02')
            blk09c = convBNReLUDrop(tf.concat([blk08, blk09a, blk09b], axis=-1), filters[6], ksize, (1,1,1), training, kprob, 1, scope='dense03')
            blk09d = convBNReLUDrop(tf.concat([blk08, blk09a, blk09b, blk09c], axis=-1), filters[6], ksize, stride, training, kprob, 1, scope='dense04')
            blk09 = convBNReLUDrop(blk09d, filters[6], ksize1, stride, training, kprob, 1, scope='dense05')

        with tf.name_scope('block10'):
            blk10 = avgpool(blk09, pool_window, pool_stride, padding='SAME', scope='s01') #1x1x1x1024

        with tf.name_scope('block11'):
            blk11a = fcn(blk10, filters[5], training, kprob, scope='s01')
            output_seg = fcn(blk11a, nseg_class, training, 1.0, scope='s04', activation=None)
    return output_seg
