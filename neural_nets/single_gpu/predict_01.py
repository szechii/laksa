'''
author:         szechi
date created:   2 Oct 2018
description:    run predict routine
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

    #predict routine
    _ = run.predict( \
        data_object     = dataset, \
        vbatchsize      = p.get('vbatchsize'), \
        out_res_path    = p.get('out_res_path'), \
        restore_path    = p.get('restore_path'))
