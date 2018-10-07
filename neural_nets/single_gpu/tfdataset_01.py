'''
author:         szechi
date created:   2 Oct 2018
description:    dataset pipeline
'''

import numpy as np
import random
import sys
import os
import tensorflow as tf
from config import parameters
p = parameters()

def cropTF(x, H, W):
    '''
    Crops or pad a [D, H, W, C] tensor to new height and new width.
    '''
    return tf.image.resize_image_with_crop_or_pad(x, H, W)

def randomAngle(deg=40): ## [-40deg,40deg] ~~--> [-0.63,0.63]
    '''
    Returns a float from [-0.63,0.63].
    '''
    return tf.random_uniform([1], minval=-deg/180.*3.142, maxval=deg/180.*3.142)

def rotateTF(x, angle):
    '''
    Rotates a [D, H, W, C] tensor in the H-W plane.
    '''
    return tf.contrib.image.rotate(x, angle)

def randomSizes(original):
    '''
    Returns a list of 2 ints from [original-25, original+55].
    '''
    return tf.cast(tf.random_uniform([2], minval=original-25, maxval=original+55), tf.int32)

def resizeTF(x, size):
    '''
    Resize function for non-binary x.
    '''
    return tf.image.resize_area(x, size)

def resizeTF1(x, size):
    '''
    Resize function for binary x.
    '''
    return tf.image.resize_nearest_neighbor(x, size)

def randomFlip(img, mask):
    '''
    Randomly flips the image and mask from left to right in H-W plane.
    Both are [D, H, W, C] tensors.
    '''
    img_out = []
    mask_out = []
    RAND = tf.random_normal([1])[0]
    def flip(img, mask):
        for i in range(img.shape[0]):
            img_out.append(tf.image.flip_left_right(img[i]))
            mask_out.append(tf.image.flip_left_right(mask[i]))
        img = tf.stack(img_out)
        mask = tf.stack(mask_out)
        return tf.cast(img, tf.float32), tf.cast(mask, tf.int32)
    def noflip(img, mask):
        return tf.cast(img, tf.float32), tf.cast(mask, tf.int32)
    img, mask = tf.cond(RAND > 0, lambda:flip(img, mask), lambda:noflip(img, mask))
    return img, mask

def TFAugmentation(img, mask, image_shape=p.get('input_data_shape')):
    '''
    Augments an image and mask.
    '''
    angle       = randomAngle()
    img         = rotateTF(img, angle)
    mask        = rotateTF(mask, angle)
    rescale     = randomSizes(image_shape[1])
    img         = cropTF(resizeTF(img, rescale), image_shape[1], image_shape[2])
    mask        = cropTF(resizeTF1(mask, rescale), image_shape[1], image_shape[2])
    img, mask   = randomFlip(img, mask)
    return img, mask

class data:
    def __init__(self):
        self.orig_size = p.get('input_data_shape')
        self.tile_size = p.get('input_data_shape')
        self.augm_size = self.tile_size
        self.seg_nclass = p.get('nclass')
        segratio = 1.0/np.array(p.get('fgbg_ratio'))
        self.loss_segweights = self.seg_nclass*segratio/np.sum(segratio)
        self.valid_batch_size = -1
        self.train_batch_size = -1
        self.batchset = { \
            'train' : p.get('TF_TRAIN_TFRECORDS'), \
            'valid' : p.get('TF_VALID_TFRECORDS'), \
            'test'  : p.get('TF_TEST_TFRECORDS')}
        self.mask_size = self.tile_size[0:-1] + (self.seg_nclass,)
        self.input_channels = self.tile_size[-1]

    def getAugmSize(self):
        return self.augm_size
    def getTileSize(self):
        return self.tile_size
    def getMaskSize(self):
        return self.mask_size
    def getMaskSize_cube(self):
        return self.mask_size_cube
    def getInputChannels(self):
        return self.input_channels
    def getOutputClasses(self):
        return self.seg_nclass
    def getValidBatchSize(self):
        return self.valid_batch_size
    def getTrainBatchSize(self):
        return self.train_batch_size
    def getLossSegWeights(self):
        return self.loss_segweights

    def readDecode(self, dataset_in, training=False):
        '''
        Example only.
        '''

        features = {'ID': tf.FixedLenFeature((), tf.string), \
                    'input': tf.FixedLenFeature((), tf.string), \
                    'mask': tf.FixedLenFeature((), tf.string)}

        parsed_features = tf.parse_single_example(dataset_in, features)

        input = tf.decode_raw(parsed_features["input"],tf.float32)
        input = tf.convert_to_tensor(input,dtype=tf.float32,)
        input = tf.reshape(input,shape=p.get('input_data_shape'))

        mask = tf.decode_raw(parsed_features["mask"],tf.int32)
        mask = tf.convert_to_tensor(mask,dtype=tf.int32,)
        mask = tf.reshape(mask,shape=p.get('input_data_shape'))

        pt = parsed_features["ID"]

        if training:
            input, mask = TFAugmentation(input, mask)

        mask1 = tf.one_hot(tf.squeeze(mask, axis=3), self.seg_nclass)

        return input, mask1, pt

    def generateBatch(self, setname, batchsize, shufflesize, training=False):
        '''
        Input Args:     setname     - 'train', 'valid', or 'test', points to \
                                    relevant tfrecords (string)
                        batchsize   - training batch size (int)
                        shufflesize - multiplier applied to training batch size \
                                    for files shuffle (int)
                        training    - true if training, otherwise false (bool)
        '''
        assert (setname in self.batchset), "setname not in batchset"
        files = self.batchset[setname]
        with tf.device('/cpu:0'):
            dataset = tf.data.TFRecordDataset(files)
            dataset = dataset.map(lambda x: self.readDecode(x, training))
            if training:
                dataset = dataset.shuffle(buffer_size=batchsize*shufflesize)
            dataset = dataset.prefetch(buffer_size=2)
            dataset = dataset.batch(batchsize)
            iterator = dataset.make_initializable_iterator()

        return iterator
