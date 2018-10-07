'''
author:         szechi
date created:   2 Oct 2018
description:    run train routine
'''

import pickle
import time
import os
import sys
import tfdataset_01 as dataset
from config import parameters
p = parameters()

assert p.get('modeltype') == 'image_segmentation_model' or \
p.get('modeltype') == 'image_classification_model', \
"Error: modeltype can be either image_segmentation_model or image_classification_model."
if p.get('modeltype') == 'image_segmentation_model':
    import run_01 as run
if p.get('modeltype') == 'image_classification_model':
    import run_02 as run

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(p.get('gpu_number'))
    dataset = dataset.data()

    #train routine
    _ = run.train( \
        data_object             = dataset, \
        readbatchsize           = p.get('readbatchsize'), \
        readshufflesize         = p.get('readshufflesize'), \
        vbatchsize              = p.get('vbatchsize'), \
        max_epochs              = p.get('max_epochs'), \
        keep_prob               = p.get('keep_prob'), \
        report_every_nsteps     = p.get('report_every_nsteps'), \
        validate_every_nepoch   = p.get('validate_every_nepoch'), \
        save_ckpt_every_nepoch  = p.get('save_ckpt_every_nepoch'), \
        save                    = p.get('save'), \
        restore                 = p.get('train_restore'), \
        restore_path            = p.get('restore_path'), \
        log_path                = p.get('log_path'))
