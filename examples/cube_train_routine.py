'''
author:         szechi
date created:   2 Oct 2018
description:    example tensorflow script to input a volume, \
                but train and validate on overlapping cubes online.
'''
import tensorflow as tf
import numpy as np
import os

def obtainPatchAndPosition(X_ph, cube_size):
    '''
    Tensorflow method of making a batch of cubes from an input volume.

    Input Args:     X_ph        - placeholder for the 3D array that will be diced \
                                (tensor of shape [None, D, H, W, C])
                    cube_size   - size of one side for all cubes (int)

    Output Args:    cubes_batch - batch of cubes (tensor of shape [N, D, H, W, C])
                    position    - list of starting z, y, x indices for each cube \
                                (tensor of shape [N, 3])
    '''
    buffer = 2 # added to avoid making a cube out of bounds from the volume array

    assert len(X_ph.shape) == 5, \
    'Error: X_ph is not of shape [None, D, H, W, C].'
    assert type(cube_size) == int, \
    'Error: cube_size is not an integer.'
    assert X_ph.shape[1].value-cube_size-buffer > 0, \
    'Error: Volume array is smaller than cube size.'

    start_z = tf.range(cube_size//2, X_ph.shape[1].value-cube_size-buffer, cube_size//2)
    start_y = tf.range(0, X_ph.shape[2].value-cube_size+1, cube_size//2)
    start_x = tf.range(0, X_ph.shape[3].value-cube_size+1, cube_size//2)
    yy, xx, zz = tf.meshgrid(start_y, start_x, start_z)
    mesh = tf.stack([zz,yy,xx], 0)
    position = tf.reshape(tf.transpose(mesh), [-1,3])

    for depth in range(X_ph.shape[1].value):
        X_slc = X_ph[:,depth,:,:,:]
        patches = tf.cast(tf.extract_image_patches(X_slc, \
                ksizes=[1,int(cube_size),int(cube_size),1], \
                strides=[1,int(cube_size//2),int(cube_size//2),1], \
                rates=[1,1,1,1], padding='VALID'), tf.float32)
        if depth == 0:
            stack = patches
        else:
            stack = tf.concat([stack,patches], axis=0)

    for i in range(start_z.shape[0].value):
        level = stack[start_z[i]:start_z[i]+cube_size,:,:,:]
        target = tf.transpose(tf.reshape(level,[cube_size,-1,cube_size,cube_size]),[1,0,2,3])
        if i == 0:
            cubes = target
        else:
            cubes = tf.concat([cubes, target], axis=0)

    cubes_batch = tf.expand_dims(cubes, axis=4)

    return cubes_batch, position

#########################################################################################################
#In this example, we have a volume of [240, 320, 320] and we will be training cubes of shape [32, 32, 32]
#Our training batchsize is 10
#Consecutive cubes have 50% overlap i.e. strides 16.
#########################################################################################################

#example only
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
volume = tf.placeholder(tf.float32,[None, 240, 320, 320, 1], name='vol_img')
all_vsl_cubes, all_vsl_pos_cubes = obtainPatchAndPosition(volume, 32)
batchsize = 10

with tf.device('/cpu:0'):
    with tf.name_scope('TFRecordCubes'):
        cube_dataset = tf.data.Dataset.from_tensor_slices(all_vsl_cubes)
        pos_dataset = tf.data.Dataset.from_tensor_slices(all_vsl_pos_cubes)
        cubepos_dataset = tf.data.Dataset.zip((cube_dataset, pos_dataset))
        cubepos_dataset = cubepos_dataset.prefetch(buffer_size=2)
        cubepos_dataset = cubepos_dataset.batch(batchsize)
        cube_iterator = cubepos_dataset.make_initializable_iterator()

with tf.Session() as sess:
    input_volume = np.ones([1, 240, 320, 320, 1], dtype=np.float32) #example only
    sess.run(cube_iterator.initializer, feed_dict={volume: input_volume})
    train_next = cube_iterator.get_next()

    while (True):
        try:
            cubes_batch, position_batch = sess.run(train_next)
        except tf.errors.OutOfRangeError:
            break

        #################################################################
        #...
        #Conduct training routine
        #...
        #################################################################

    #####################################################################
    #...
    #Put back all cubes to their original location in the volume based on
    #position_batch
    #...
    #####################################################################
