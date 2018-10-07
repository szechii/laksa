'''
author:         szechi
date created:   2 Oct 2018
description:    class to do skull stripping on TOF nii 3D float image
'''

#######################################
#example use
#skullstrip("tof.nii.gz").removeskull()
#######################################

import scipy.stats as sp
import itk
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab

class skullstrip():
    '''
    Approximate skullstripping for a TOF image volume by morphological erosion.

    Input Args:     img_path - file directory of the TOF nii image (string)

    Output Args:    itk_obj  - TOF with skull approximately removed (itk.F image)
    '''
    def __init__(self, img_path):

        assert os.path.exists(img_path), \
        print ('Error: img_path does not exist.')

        self.img_path = img_path

    def removeskull(self):
        #read the image
        pixelType = itk.F
        imageDimesion = 3
        imageType = itk.Image[pixelType, imageDimesion]
        itk_py_converter = itk.PyBuffer[imageType]
        reader = itk.ImageFileReader[imageType].New()
        reader.SetFileName(self.img_path)
        reader.Update()
        reader_np = itk_py_converter.GetArrayFromImage(reader.GetOutput())
        #derive background intensity upper limit
        intensities = np.ndarray.flatten(reader_np)
        (mu, sigma) = norm.fit(intensities)
        n, bins = np.histogram(intensities, 100, normed=True)
        y = mlab.normpdf( bins, mu, sigma)
        int_freq = np.array(sp.itemfreq(intensities),dtype=int)
        int_upthresh = int(np.argmin(int_freq[0:int(bins[np.argmax(y)]),1]))
        #padding a border to image
        padFilter = itk.ConstantPadImageFilter[imageType, imageType].New()
        size = itk.Size[3]()
        size[0] = 10
        size[1] = 10
        padFilter.SetPadLowerBound(size)
        padFilter.SetPadUpperBound(size)
        padFilter.SetConstant(0)
        padFilter.SetInput(reader.GetOutput())
        padFilter.Update()
        #obtain binary mask of head
        thresholdFilter = itk.BinaryThresholdImageFilter[imageType, imageType].New()
        thresholdFilter.SetInput(padFilter.GetOutput())
        thresholdFilter.SetLowerThreshold(int_upthresh)
        thresholdFilter.SetUpperThreshold(int(np.max(itk_py_converter.GetArrayFromImage(reader.GetOutput()))))
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.Update()
        #define types
        OutputImageType = itk.Image[itk.F, 3]
        InputImageType = itk.Image[itk.UC, 3]
        #cast to itk.UC
        castImageFilter = itk.CastImageFilter[imageType, InputImageType].New()
        castImageFilter.SetInput(thresholdFilter.GetOutput())
        castImageFilter.Update()
        cast_np = itk.GetArrayFromImage(castImageFilter.GetOutput())
        #holefilling for closing
        holeFilter = itk.VotingBinaryIterativeHoleFillingImageFilter[itk.Image[itk.UC,2]].New()
        holeFilter.SetRadius(15)
        holeFilter.SetMajorityThreshold(1)
        holeFilter.SetForegroundValue(1)
        holeFilter.SetMaximumNumberOfIterations(3)
        slicer = itk.SliceBySliceImageFilter[InputImageType,InputImageType].New()
        slicer.SetInput(castImageFilter.GetOutput())
        slicer.SetFilter(holeFilter)
        slicer.Update()
        #morphological closing
        StructuringElementType1 = itk.FlatStructuringElement[2]
        structuringElement = StructuringElementType1.Ball(50)
        closeFilter = itk.BinaryMorphologicalClosingImageFilter[itk.Image[itk.UC,2],itk.Image[itk.UC,2],StructuringElementType1].New()
        closeFilter.SetForegroundValue(1)
        closeFilter.SetKernel(structuringElement)
        slicer1 = itk.SliceBySliceImageFilter[InputImageType,InputImageType].New()
        slicer1.SetInput(slicer.GetOutput())
        slicer1.SetFilter(closeFilter)
        slicer1.Update()
        #final holefilling to make sure binary mask is completely filled
        hole = itk.BinaryFillholeImageFilter[itk.Image[itk.UC,2]].New()
        hole.SetForegroundValue(1)
        slicer2 = itk.SliceBySliceImageFilter[InputImageType,InputImageType].New()
        slicer2.SetInput(slicer1.GetOutput())
        slicer2.SetFilter(hole)
        slicer2.Update()
        #erosion
        structuringElement = StructuringElementType1.Ball(40)
        ErodeFilterType = itk.BinaryErodeImageFilter[itk.Image[itk.UC,2],itk.Image[itk.UC,2], StructuringElementType1]
        erodeFilter = ErodeFilterType.New()
        erodeFilter.SetKernel(structuringElement)
        erodeFilter.SetForegroundValue(1)
        erodeFilter.SetBackgroundValue(0)
        slicer3 = itk.SliceBySliceImageFilter[InputImageType,InputImageType].New()
        slicer3.SetInput(slicer2.GetOutput())
        slicer3.SetFilter(erodeFilter)
        slicer3.Update()
        #skull = original - eroded
        subtractFilter = itk.SubtractImageFilter[InputImageType,InputImageType,InputImageType].New()
        subtractFilter.SetInput1(slicer2.GetOutput())
        subtractFilter.SetInput2(slicer3.GetOutput())
        subtractFilter.Update()
        #remove padding by crop
        cropFilter = itk.CropImageFilter[InputImageType,InputImageType].New()
        cropFilter.SetInput(subtractFilter.GetOutput())
        cropFilter.SetBoundaryCropSize(size)
        cropFilter.Update()
        #invert mask
        invertFilter = itk.InvertIntensityImageFilter[InputImageType, InputImageType].New()
        invertFilter.SetInput(cropFilter.GetOutput())
        invertFilter.SetMaximum(1)
        invertFilter.Update()
        #mask the image
        maskFilter = itk.MaskImageFilter[OutputImageType,InputImageType,OutputImageType].New()
        maskFilter.SetInput(reader.GetOutput())
        maskFilter.SetMaskImage(invertFilter.GetOutput())
        maskFilter.Update()
        mask_np = itk.GetArrayFromImage(maskFilter.GetOutput())
        #return the itk object
        itkobj = itk.GetImageFromArray(mask_np)
        itkobj.SetMetaDataDictionary(reader.GetOutput().GetMetaDataDictionary())
        itkobj.SetDirection(reader.GetOutput().GetDirection())
        itkobj.SetSpacing(reader.GetOutput().GetSpacing())
        itkobj.SetOrigin(reader.GetOutput().GetOrigin())

        return itkobj
