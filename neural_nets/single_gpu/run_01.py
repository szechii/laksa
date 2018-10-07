'''
author:         szechi
date created:   2 Oct 2018
description:    train and validation script for image segmentation (3D)
'''

import sys
import numpy as np
import tensorflow as tf
import scipy.ndimage
sys.path.append('../../utils')
import helper as helper
import costfunction as costfunction
from costaccumulator import costAccumulator
import model_01 as model
from PIL import Image
from config import parameters
p = parameters()

def train(data_object, readbatchsize=1, readshufflesize=1, vbatchsize=1, \
    max_epochs=1000, keep_prob=1.0, report_every_nsteps=0, validate_every_nepoch=0, \
    save_ckpt_every_nepoch=0, save=False, restore=False, \
    restore_path='tfgraph', log_path='logdirectory'):

    '''
    Train routine

    Input Args:     data_object             - module from data class in tfdataset_01.py (module)
                    readbatchsize           - training batch size (int)
                    readshufflesize         - multiplier applied to training batch size \
                                            for files shuffle (int)
                    vbatchsize              - validation batch size (int)
                    max_epochs              - number of epochs to train (int)
                    keep_prob               - keep probability for dropout (float)
                    report_every_nsteps     - print batch training result after every \
                                            certain number of steps (int)
                    validate_every_nepoch   - number of train epochs before every valdation routine (int)
                    save_out_every_nepoch   - save out model checkpoint for every certain number of epochs (int)
                    save                    - checkpoint will be saved after training the last epoch \
                                            if set true, and otherwise if false (bool)
                    restore                 - restores checkpoint from restore_path before train routine
                    restore_path            - file directory of the checkpoint to restore
                    log_path                - folder directory to save outputs

    Output Args:    None
    '''

    assert readbatchsize >0,                    "Error: batchsize must be >0"
    assert readshufflesize >0,                  "Error: readshufflesize must be >0"
    assert vbatchsize >0,                       "Error: vbatchsize must be >0"
    assert (keep_prob >0 and keep_prob<=1.0),   "Error: kprob must be between (0,1.0]"
    assert report_every_nsteps >=0,             "Error: report_every must be >=0"
    assert validate_every_nepoch >=0,           "Error: validate_every must be >=0"
    assert save_ckpt_every_nepoch >=0,          "Error: save_ckpt must be >=0"
    assert save == True or save == False,       "Error: save must be boolean"
    assert restore == True or restore == False, "Error: restore must be boolean"


    tf.reset_default_graph()

    # Read & decode tfrecord
    print "Read & Decode"
    sys.stdout.flush()
    with tf.device('/cpu:0'):
        with tf.name_scope('TFRecord'):
            train_iterator = data_object.generateBatch('train', \
                readbatchsize, readshufflesize, training=True)
            valid_iterator = data_object.generateBatch('valid', \
                vbatchsize, readshufflesize, training=False)

    # Define input & output placeholders into model
    masksize = data_object.getMaskSize()
    nclass = data_object.getOutputClasses()

    # Seg weights squared
    sweights = np.asarray(data_object.getLossSegWeights())
    sweights2 = sweights*sweights
    sweights2 = 1.0*sweights2/np.sum(sweights2)
    dd, hh, ww, cc = p.get('input_data_shape')

    with tf.name_scope('InputPlaceHolder'):
        tof_in = tf.placeholder(tf.float32,[None, dd, hh, ww, cc], name='tof_in')
        y_aseg = tf.placeholder(tf.int32,[None, dd, hh, ww, nclass], name='y_aseg')
        case_in = tf.placeholder(tf.string)

        track_loss = tf.placeholder(tf.float32)
        track_acc = tf.placeholder(tf.float32)
        track_sftmax = tf.placeholder(tf.float32)
        track_dice = tf.placeholder(tf.float32)

        training = tf.placeholder(tf.bool) # T/F for batchnorm
        kprob = tf.placeholder(tf.float32) # Dropout prob

    print "Declare model"
    # Output from model
    y_pseg = model.image_segmentation_model(tof_in, masksize, training, kprob)
    y_psfmax = tf.nn.softmax(y_pseg, dim=-1)
    sys.stdout.flush()

    # Training vars
    train_accu_seg = costfunction.fg_accuracy(y_aseg, y_pseg, axis=-1)
    train_sfmax_seg = costfunction.weightedFocalCrossEntropyLoss(y_aseg, y_pseg, \
        weighting=True, weights=sweights, focal=True, gamma=2)
    train_sdice_seg = costfunction.weightedDiceLoss(y_aseg, y_pseg, \
        invarea=False, weighting=True, weights=sweights2, hard=False, gamma=2)
    train_cost = 10*train_sdice_seg + 1000*train_sfmax_seg


    tracking_loss = tf.summary.scalar('total_cost', track_loss)
    tracking_acc = tf.summary.scalar('accuracy', track_acc)
    tracking_sftmax = tf.summary.scalar('softmax', track_sftmax)
    tracking_dice = tf.summary.scalar('dice', track_dice)

    # Minimizer vars
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.99, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost, global_step=global_step)

    # Object to accumulate costs etc.
    train_accumulator = costAccumulator(["train_cost", "train_accu", "train_sfmax", "train_sdice"])
    valid_accumulator = costAccumulator(["valid_cost", "valid_accu", "valid_sfmax", "valid_sdice"])

    print "Start session"
    sys.stdout.flush()
    f_train = open(log_path + "train_log.txt", "w+")
    f_valid = open(log_path + "valid_log.txt", "w+")
    print "Log files created"
    sys.stdout.flush()
    # valid loop

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ImageSegmentationModel))
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(log_path + '/valid', sess.graph)
        # Initialize tf variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        '''
        # Check model variables
        model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
        for ivar in model_variables:
            print ivar.name
        '''
        if restore:
            saver.restore(sess, restore_path)
            print "Restored graph from " + restore_path

        iepoch = 0
        istep = 0

        while (iepoch < max_epochs):
            istep = 0
            train_accumulator.resetCost()

            print "EPOCH %5d" % iepoch

            sess.run(train_iterator.initializer)
            train_next = train_iterator.get_next()

            # Training batch loop
            while (True):
                # Fetch batch from tfdataset and shuffle
                try:
                    tof_in_sb, y_aseg_sb, pt = sess.run(train_next)
                except tf.errors.OutOfRangeError:
                    break

                feedDict = { \
                    tof_in: tof_in_sb, \
                    y_aseg: y_aseg_sb, \
                    training: True, kprob: keep_prob}

                _, cost1, cost2, cost3, cost4, \
                lrn_sb, gstep = sess.run([optimizer, \
                    train_cost, \
                    train_accu_seg, \
                    train_sfmax_seg, \
                    train_sdice_seg, \
                    learning_rate, global_step], \
                    feed_dict=feedDict)

                train_accumulator.addCost('train_cost', cost1)
                train_accumulator.addCost('train_accu', cost2)
                train_accumulator.addCost('train_sfmax', cost3)
                train_accumulator.addCost('train_sdice', cost4)
                train_accumulator.addDivisorCount()

                if (report_every_nsteps > 0 and istep % report_every_nsteps == 0):
                    print "Epoch: %5d Step: %5d gStep: %5d alpha: %7.10f %s" % \
                        (iepoch, istep, gstep, lrn_sb, train_accumulator.strNewCost())
                    sys.stdout.flush()

                istep += 1

            train_accumulator.makeMean()

            tensorboard_feeder = {track_acc:train_accumulator.getValues()[3], \
                                track_loss:train_accumulator.getValues()[0], \
                                track_sftmax:train_accumulator.getValues()[1], \
                                track_dice:train_accumulator.getValues()[2]}
            log1, log2, log3, log4 = sess.run([tracking_acc, tracking_loss, tracking_sftmax, \
                                    tracking_dice], feed_dict=tensorboard_feeder)
            train_writer.add_summary(log1, iepoch)
            train_writer.flush()
            train_writer.add_summary(log2, iepoch)
            train_writer.flush()
            train_writer.add_summary(log3, iepoch)
            train_writer.flush()
            train_writer.add_summary(log4, iepoch)
            train_writer.flush()

            print "\tiepoch: %5d %s" % (iepoch, train_accumulator.strCost('epoch_'))
            f_train.write("\tiepoch: %5d %s" % (iepoch, train_accumulator.strCost('epoch_')))
            sys.stdout.flush()

            if iepoch % save_ckpt_every_nepoch == 0:
                log_path_return = saver.save(sess, log_path+'epoch'+str(iepoch)+'_tfgraph.ckpt')
                print "Saved graph at step ",istep, " in " + log_path_return

            if (validate_every_nepoch > 0 and iepoch % validate_every_nepoch == 0) or (iepoch == max_epochs-1):
                # Validation batch loop
                sess.run(valid_iterator.initializer)
                valid_next = valid_iterator.get_next()

                valid_accumulator.resetCost()
                while (True):
                    # Fetch batch from tfdataset and shuffle
                    try:
                        tof_in_sb, y_aseg_sb, pt = sess.run(valid_next)
                    except tf.errors.OutOfRangeError:
                        break

                    feedDict = { \
                        tof_in: tof_in_sb, \
                        y_aseg: y_aseg_sb, \
                        training: False, kprob: 1.0}
                    cost1, cost2, cost3, cost4, \
                    y_aseg_sb, y_pred_sb, image_= sess.run([ \
                        train_cost, \
                        train_accu_seg, \
                        train_sfmax_seg, \
                        train_sdice_seg, \
                        y_aseg, \
                        y_psfmax, \
                        tof_in], feed_dict=feedDict)

                    valid_accumulator.addCost('valid_cost', cost1)
                    valid_accumulator.addCost('valid_accu', cost2)
                    valid_accumulator.addCost('valid_sfmax', cost3)
                    valid_accumulator.addCost('valid_sdice', cost4)
                    valid_accumulator.addDivisorCount()

                    heatList.append(y_pred_sb[0,:,:,:,1])
                    actualList.append(np.squeeze(np.argmax(y_aseg_sb, axis=4)))

                valid_accumulator.makeMean()

                tensorboard_feeder = {track_acc:valid_accumulator.getValues()[3], \
                                    track_loss:valid_accumulator.getValues()[0], \
                                    track_sftmax:valid_accumulator.getValues()[1], \
                                    track_dice:valid_accumulator.getValues()[2]}
                log1, log2, log3, log4 = sess.run([tracking_acc, tracking_loss, tracking_sftmax, \
                                        tracking_dice], feed_dict=tensorboard_feeder)
                valid_writer.add_summary(log1, iepoch)
                valid_writer.flush()
                valid_writer.add_summary(log2, iepoch)
                valid_writer.flush()
                valid_writer.add_summary(log3, iepoch)
                valid_writer.flush()
                valid_writer.add_summary(log4, iepoch)
                valid_writer.flush()

                print "\tiepoch: %5d %s" % (iepoch, valid_accumulator.strCost('epoch_'))
                sys.stdout.flush()
                f_valid.write("\tiepoch: %5d %s" % (iepoch, valid_accumulator.strCost('epoch_')))

                #save best accuracy epoch
                if iepoch == 0:
                    tag = valid_accumulator.getValues()[3]
                    print "Saved best: epoch" + str(iepoch) + ' '+ log_path +'bestcheckpoint.ckpt'
                    saver.save(sess, log_path+'bestcheckpoint.ckpt')
                else:
                    if valid_accumulator.getValues()[3] > tag:
                        print "Saved best: epoch" + str(iepoch) + ' '+ log_path +'bestcheckpoint.ckpt'
                        saver.save(sess, log_path+'bestcheckpoint.ckpt')
                        tag = valid_accumulator.getValues()[3]

            iepoch += 1

        # Save final step of the epoch
        if save:
            log_path_return = saver.save(sess, log_path)
            print "Saved final graph in " + log_path_return

    try:
        sess.close()
    except:
        print "Unable to close session"

    return 0


def predict(data_object, vbatchsize=1, out_res_path=None, restore_path='tfgraph'):
    '''
    Predict routine

    Input Args:     data_object     - module from data class in tfdataset_01.py (module)
                    vbatchsize      - validation batch size (int)
                    out_res_path    - folder directory to save outputs
                    restore_path    - file directory of the checkpoint to restore

    Output Args:    None
    '''


    assert vbatchsize > 0, "Error: vbatchsize must be >0"
    tf.reset_default_graph()

    # Read & decode tfrecord
    print "Read & Decode"
    sys.stdout.flush()
    with tf.device('/cpu:0'):
        with tf.name_scope('TFRecord'):
            valid_iterator = data_object.generateBatch('valid', \
                vbatchsize, vbatchsize, training=False)

    # Define input & output placeholders into model
    masksize = data_object.getMaskSize()
    nclass = data_object.getOutputClasses()
    # seg weights squared
    sweights = np.asarray(data_object.getLossSegWeights())
    sweights2 = sweights*sweights
    sweights2 = 1.0*sweights2/np.sum(sweights2)
    dd, hh, ww, cc = p.get('input_data_shape')

    with tf.name_scope('InputPlaceHolder'):
        tof_in = tf.placeholder(tf.float32,[None, dd, hh, ww, cc], name='tof_in')
        y_aseg = tf.placeholder(tf.int32,[None, dd, hh, ww, nclass], name='y_aseg')
        case_in = tf.placeholder(tf.string)

        training = tf.placeholder(tf.bool) # T/F for batchnorm
        kprob = tf.placeholder(tf.float32) # Dropout prob

    print "Declare model"
    # Output from model
    y_pseg = model.image_segmentation_model(tof_in, masksize, training, kprob)
    y_psfmax = tf.nn.softmax(y_pseg, dim=-1)
    sys.stdout.flush()

    # Training vars
    train_accu_seg = costfunction.fg_accuracy(y_aseg, y_pseg, axis=-1)
    train_sfmax_seg = costfunction.weightedFocalCrossEntropyLoss(y_aseg, y_pseg, \
        weighting=True, weights=sweights, focal=True, gamma=2)
    train_sdice_seg = costfunction.weightedDiceLoss(y_aseg, y_pseg, \
        invarea=False, weighting=True, weights=sweights2, hard=False, gamma=2)
    train_cost = 10*train_sdice_seg + 1000*train_sfmax_seg

    # Object to accumulate costs etc.
    valid_accumulator = costAccumulator(["test_cost", "test_accu", "test_sfmax", "test_sdice"])

    print "Start session"
    sys.stdout.flush()


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        # Initialize tf variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        '''
        # Check model variables
        model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
        for ivar in model_variables:
            print ivar.name
        '''
        saver = tf.train.Saver()
        saver.restore(sess, restore_path)
        print "Restored graph from " + restore_path

        # Validation batch loop
        sess.run(valid_iterator.initializer)
        valid_next = valid_iterator.get_next()

        istep = 0
        heatList = []
        actualList = []
        ptList = []

        valid_accumulator.resetCost()
        while (True):
            # Fetch batch from tfdataset and shuffle
            try:
                tof_in_sb, y_aseg_sb, pt = sess.run(valid_next)
            except tf.errors.OutOfRangeError:
                break

            # Split fetched batch into smaller training samples
            feedDict = { \
                tof_in: tof_in_sb, \
                y_aseg: y_aseg_sb, \
                training: False, kprob: 1.0}

            cost1, cost2, cost3, cost4, \
            y_aseg_sb, y_pred_sb, image_= sess.run([ \
                train_cost, \
                train_accu_seg, \
                train_sfmax_seg, \
                train_sdice_seg, \
                y_aseg, \
                y_psfmax, \
                tof_in], feed_dict=feedDict)

            valid_accumulator.addCost('test_cost', cost1)
            valid_accumulator.addCost('test_accu', cost2)
            valid_accumulator.addCost('test_sfmax', cost3)
            valid_accumulator.addCost('test_sdice', cost4)
            valid_accumulator.addDivisorCount()

            heatList.append(y_pred_sb[0,:,:,:,1])
            actualList.append(np.squeeze(np.argmax(y_aseg_sb, axis=4)))
            ptList.append(pt[0])

            istep += 1

        #finding tp, fp
        thresh_test = np.round(np.arange(0.0,1.1,0.05),2)
        for k in range(len(thresh_test)):
            TP, FP, FN = costfunction.eval_metrics(heatList, actualList, thresh_test[k])
            print(str(thresh_test[k])+' '+str(TP)+' '+str(FP)+' '+str(FN))

        valid_accumulator.makeMean()
        print "\tOverall %s" % (valid_accumulator.strCost(''))
        sys.stdout.flush()

    try:
        sess.close()
    except:
        print "Unable to close session"

    return 0
