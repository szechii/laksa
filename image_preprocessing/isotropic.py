'''
author:         szechi
date created:   2 Oct 2018
description:    function that makes an image array with poorer resolution \
                in the z-direction than in the x/y-direction, isotropic
'''

import nibabel as nib
import numpy as np
import scipy.ndimage

def makeIsotropic(nii_directory):
    '''
    Makes an image array isotropic.

    Input Args:     nii_directory   - nii image directory (string)

    Output Args:    tof_isotropic   - isotropic 3D image array (np.array)
    '''
    assert os.path.exists(nii_directory), 'Error: nii file does not exist.'

    meta = nib.load(nii_directory)
    header = meta.header
    tof_arr = meta.get_data()
    tof_zooms = header.get_zooms()
    scale = round(tof_zooms[0]/tof_zooms[2],1)
    tof_isotropic = scipy.ndimage.interpolation.zoom(tof_arr,[scale, scale,1])

    return tof_isotropic
