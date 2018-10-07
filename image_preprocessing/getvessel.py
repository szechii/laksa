'''
author:         szechi
date created:   2 Oct 2018
description:    class to do vessel segmentation after skull stripping
'''

#######################################
#example use
#vesselsegmentation("tof.nii.gz").getnii()
#######################################

import itk
import numpy as np
import scipy.stats
import nibabel as nib
from skullstripping import skullstrip

def cutoffs(vesselness_np, smooth_np):
    max_vesselness = np.max(vesselness_np)
    level = 0.25*max_vesselness
    sel_mask = vesselness_np >= level
    sel_vox = np.extract(sel_mask,smooth_np)
    mean = np.mean(sel_vox)
    std = np.std(sel_vox)
    med = np.median(sel_vox)
    skew = scipy.stats.skew(sel_vox)
    if skew > 0:
        ref_val = min(med,mean)
    else:
        ref_val = max(med,mean)
    val_cutoff = ref_val-1.5*std
    return (val_cutoff, level)

def get_dicom_affine(ref):
    gridSize = ref.GetLargestPossibleRegion().GetSize()
    nDim = ref.GetImageDimension()
    vSpacing = itk.GetArrayFromVnlVector(ref.GetSpacing().GetVnlVector())
    mDirection = ref.GetDirection().GetVnlMatrix()
    vOrigin = itk.GetArrayFromVnlVector(ref.GetOrigin().GetVnlVector())
    dmat = np.zeros((nDim+1,nDim+1))
    for i in range(0,nDim):
        for j in range(0,nDim):
            dmat[i,j] = mDirection(i,j)
    dmat[0:nDim,0:nDim] = dmat[0:nDim,0:nDim]*vSpacing
    dmat[0:nDim,nDim] = vOrigin.transpose()
    dmat[nDim,:] = [0,0,0,1]
    return dmat

class vesselsegmentation():
    '''
    Performs vessel segmentation by region growing from randomly selected seeds \
    in a binary hessian vesselness mask and thresholded time-of-flight (TOF) image, \
    then saves it as a nifty object.

    Input Args:     img_path    - file directory of the TOF nii image (string)
                    save_path   - file directory to save vessel segmentation as nii (string)

    Output Args:    None
    '''
    def __init__(self, img_path, save_path):

        assert os.path.exists(img_path), \
        print ('Error: img_path does not exist.')
        assert os.path.exists(os.path.dirname(save_path)), \
        print ('Error: parent directory of img_path does not exist.')
        assert save_path[-7:] == '.nii.gz' or save_path[-4:] == '.nii', \
        print ('Error: save_path is not *.nii.gz or *.nii.')

        self.img_path = img_path
        self.save_path = save_path

    def getnii(self):
        # Read TOF nii
        pixelType = itk.F
        imageDimesion = 3
        imageType = itk.Image[pixelType, imageDimesion]
        itk_py_converter = itk.PyBuffer[imageType]
        reader = itk.ImageFileReader[imageType].New()
        reader.SetFileName(self.img_path)
        reader.Update()
        #remove skull
        img_noskull = skullstrip(self.img_path).removeskull()
        # Rescale intensity
        intensity_filter = itk.RescaleIntensityImageFilter[imageType,imageType].New()
        intensity_filter.SetInput(img_noskull)
        intensity_filter.SetOutputMinimum(0)
        intensity_filter.SetOutputMaximum(1000)
        # Smoothing
        smoothing_filter = itk.CurvatureFlowImageFilter[imageType,imageType].New()
        smoothing_filter.SetInput(intensity_filter.GetOutput())
        smoothing_filter.SetNumberOfIterations(1)
        smoothing_filter.SetTimeStep(0.05)
        smoothing_filter.Update()
        smooth_np = itk_py_converter.GetArrayFromImage(smoothing_filter.GetOutput())
        #histeq filter to original image
        histeq_filter = itk.AdaptiveHistogramEqualizationImageFilter[imageType].New()
        histeq_filter.SetInput(img_noskull)
        histeq_filter.SetAlpha(0.5)
        histeq_filter.SetBeta(0.5)
        radius = itk.Size[img_noskull.GetImageDimension()]()
        radius.Fill(5)
        histeq_filter.SetRadius(radius)
        histeq_filter.Update()
        # Rescale intensity of histeq
        intensity_filter1 = itk.RescaleIntensityImageFilter[imageType,imageType].New()
        intensity_filter1.SetInput(histeq_filter.GetOutput())
        intensity_filter1.SetOutputMinimum(0)
        intensity_filter1.SetOutputMaximum(1000)
        # Smoothing
        smoothing_filter1 = itk.CurvatureFlowImageFilter[imageType,imageType].New()
        smoothing_filter1.SetInput(intensity_filter1.GetOutput())
        smoothing_filter1.SetNumberOfIterations(10)
        smoothing_filter1.SetTimeStep(0.02)
        smoothing_filter1.Update()
        smooth_np1 = itk_py_converter.GetArrayFromImage(smoothing_filter1.GetOutput())
        # Hessian with Gaussian Smoothing
        hessian_filter1 = itk.HessianRecursiveGaussianImageFilter[imageType].New()
        hessian_filter1.SetSigma(1.0)
        hessian_filter1.SetInput(smoothing_filter1.GetOutput())
        # Vesselness based on Hessian Eigenvector closest to 0
        vesselness_filter1 = itk.Hessian3DToVesselnessMeasureImageFilter[pixelType].New()
        vesselness_filter1.SetInput(hessian_filter1.GetOutput())
        vesselness_filter1.Update()
        vesselness_np1 = itk_py_converter.GetArrayFromImage(vesselness_filter1.GetOutput())
        #valcutoff and level
        val_cutoff, level = cutoffs(vesselness_np1, smooth_np)
        # Threshold x vesselness
        image_thresh = np.swapaxes(vesselness_np1 > level*2.2,0,2)*np.swapaxes(smooth_np > val_cutoff,0,2)
        # Confidence Connected Routine
        imageTypeOut = itk.Image[itk.UC, 3]
        connect_filter = itk.ConfidenceConnectedImageFilter[imageType,imageTypeOut].New()
        connect_filter.SetInput(histeq_filter.GetOutput())
        connect_filter.SetMultiplier(1.0)
        connect_filter.SetNumberOfIterations(15) #do not need so many iterations for good vessel structure, shorten run time too
        connect_filter.SetReplaceValue(1)
        # Grow from sample of points
        image_index = np.concatenate((np.indices(image_thresh.shape),np.expand_dims(image_thresh,axis=0)),axis=0)
        image_index = np.reshape(image_index,(4,-1))
        image_index = image_index[:,np.where(image_index[-1,:] > 0)[0]]
        image_index = np.transpose(image_index)
        np.random.shuffle(image_index)
        image_sel = image_index[0:int(300),:]
        for point in image_sel:
            itkIndex = itk.Index[3]()
            itkIndex[0] = int(point[0])
            itkIndex[1] = int(point[1])
            itkIndex[2] = int(point[2])
            connect_filter.AddSeed(itkIndex)
        connect_filter.SetInitialNeighborhoodRadius(1)
        connect_filter.Update()
        connect_np = itk.GetArrayFromImage(connect_filter.GetOutput())
        #save to nii
        dicom_affine = get_dicom_affine(reader.GetOutput())
        nii_mask = nib.Nifti1Image(np.swapaxes(connect_np,0,2),dicom_affine)
        nib.save(nii_mask,self.save_path)

        return 0
