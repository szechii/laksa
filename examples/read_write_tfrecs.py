'''
author:         szechi
date created:   5 Oct 2018
description:    example functions to write tfrecords and read iteratively
'''

import tensorflow as tf
import os
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int32_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int32List(value=[value]))

def write_tfrecords(float_arr, int_arr, save_dir):
    '''
    Example routine to write tfrecords. In this function, we will be writing \
    one example consisting of a float array, int array and a string into a tfrecord.

    Input Args:     float_arr   - float array (np.array)
                    int_arr     - int array (np.array)
                    save_dir    - *.tfrecords file directory to write (string)

    Output Args:    None
    '''
    assert float_arr.dtype == np.float32, \
    print ('Error: float_arr is not a float32 np.array.')
    assert int_arr.dtype == np.int32, \
    print ('Error: int_arr is not a int32 np.array.')
    assert save_dir[-3:] == '.tf' or save_dir[-10:] == '.tfrecords', \
    print ('Error: save_dir is not a .tfrecords or .tf file.')
    assert os.path.exists(os.path.dirname(save_dir)), \
    print ('Error: parent directory of save_dir does not exist.')

    writer = tf.python_io.TFRecordWriter(save_dir)
    print('writing: '+save_dir)
    input_raw = np.array(float_arr, dtype=np.float32).tostring()
    label_raw = np.array(int_arr, dtype=np.int32).tostring()
    string_raw = 'example_only'
    example = tf.train.Example(features=tf.train.Features(feature={
        'input': _bytes_feature(input_raw),
        'label': _bytes_feature(label_raw),
        'example_string': _bytes_feature(str(string_raw).encode('utf-8'))}))

    writer.write(example.SerializeToString())
    writer.close()

    return 0

def read_tfrecords(tfrecord_dir, data_shape):
    '''
    Example routine to read one example in a tfrecord. In this function, we will be \
    reading a tfrecord that has a float feature 'input' and int feature 'label'.

    Input Args:     tfrecord_dir    - *.tfrecords file directory to read (string)
                    data_shape      - N-tuple indicating shape of the N-dimensional \
                                    input and label (N-tuple)

    Output Args:    None
    '''

    assert os.path.exists(tfrecord_dir), \
    print ('Error: tfrecord_dir does not exist.')

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_dir)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        input = (example.features.feature['input'].bytes_list.value[0])
        label = (example.features.feature['label'].bytes_list.value[0])
        input = np.fromstring(input, dtype=np.float32).reshape(data_shape)
        label = np.fromstring(label, dtype=np.int32).reshape(data_shape)

    return 0
