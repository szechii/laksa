'''
author:         szechi
date created:   2 Oct 2018
description:    function that calculates hessian vesselness
'''

import os
import numpy as np
import itk

def veshness(array):
    '''
    Returns hessian vesselness of a 3D float array.

    Input Args:     array           - 3D array which hessian vesselness will be calculated (np.array)

    Output Args:    vesselness_np   - 3D hessian vesselness of the input array (np.array)
    '''
    assert len(array.shape) == 3,   'Error: array is not a 3D array.'

    img = itk.GetImageFromArray(np.array(array, np.float32))
    pixelType = itk.F
    imageDimesion = 3
    imageType = itk.Image[pixelType, imageDimesion]
    itk_py_converter = itk.PyBuffer[imageType]
    # Smoothing
    smoothing_filter = itk.CurvatureFlowImageFilter[imageType,imageType].New()
    smoothing_filter.SetInput(img)
    smoothing_filter.SetNumberOfIterations(10)
    smoothing_filter.SetTimeStep(0.02)
    smoothing_filter.Update()
    smooth_np1 = itk_py_converter.GetArrayFromImage(smoothing_filter.GetOutput())
    # Hessian with Gaussian Smoothing
    hessian_filter = itk.HessianRecursiveGaussianImageFilter[imageType].New()
    hessian_filter.SetSigma(1.0)
    hessian_filter.SetInput(smoothing_filter.GetOutput())
    # Vesselness based on Hessian Eigenvector closest to 0
    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[pixelType].New()
    vesselness_filter.SetInput(hessian_filter.GetOutput())
    vesselness_filter.Update()
    vesselness_np = itk_py_converter.GetArrayFromImage(vesselness_filter.GetOutput())

    return vesselness_np
