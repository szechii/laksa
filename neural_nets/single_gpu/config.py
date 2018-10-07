'''
author:         szechi
date created:   2 Oct 2018
description:    config file for input arguments (3D neural net)
'''

import os
import sys

class parameters:
    '''
    To run train and predict routine, these input parameters have to be updated.

    Input Args: gpu_number              - gpu ID as in os.environ["CUDA_VISIBLE_DEVICES"] (int)
                modeltype               - type of model to use, 'image_segmentation_model' or 'image_classification_model'
                input_data_shape        - input data shape [D, H, W, C] (4-tuple)
                nclass                  - number of classes (int)
                fgbg_ratio              - ratio of occurence across classes (nclass-tuple)
                readbatchsize           - training batch size (int)
                readshufflesize         - multiplier applied to training batch size \
                                        for files shuffle (int)
                vbatchsize              - validation batch size (int)
                max_epochs              - number of epochs to train (int)
                keep_prob               - keep probability for dropout (float)
                report_every_nsteps     - print batch training result after every \
                                        certain number of steps (int)
                validate_every_nepoch   - number of train epochs before every valdation routine (int)
                save_ckpt_every_nepoch  - save out model checkpoint for every certain number of epochs (int)
                save                    - checkpoint will be saved after training the last epoch \
                                        if set true, and otherwise if false (bool)
                train_restore           - restores checkpoint from restore_path before train routine
                restore_path            - file directory of the checkpoint to restore
                log_path                - folder directory to save outputs
                train_list              - list of tfrecords for train
                valid_list              - list of tfrecords for validation
                test_list               - list of tfrecords for test
    '''

    def __init__(self, out_res_path, restore_path, log_path, train_list, \
        valid_list, test_list):
        self.out_res_path = out_res_path
        self.restore_path = restore_path
        self.log_path = log_path
        self.train_list = train_list
        self.valid_list = valid_list
        self.test_list = test_list

        __dict = { \
        'gpu_number':               5, \
        'modeltype':                "image_classification_model", \
        'input_data_shape':         (88, 80, 80, 2), \
        'nclass':                   2, \
        'fgbg_ratio':               (0.999765, 0.000235), \
        'readbatchsize':            2, \
        'readshufflesize':          5, \
        'vbatchsize':               1, \
        'max_epochs':               602, \
        'keep_prob':                0.7, \
        'report_every_nsteps':      1, \
        'validate_every_nepoch':    1, \
        'save_ckpt_every_nepoch':   50, \
        'out_res_path':             self.out_res_path, \
        'save':                     True, \
        'train_restore':            False, \
        'restore_path':             self.restore_path, \
        'log_path':                 self.log_path, \
        'TF_TRAIN_TFRECORDS':       self.train_list, \
        'TF_VALID_TFRECORDS':       self.valid_list, \
        'TF_TEST_TFRECORDS':        self.test_list, \
        }
        self.__dict = __dict

    def get(self, key):
        return self.__dict[key]
