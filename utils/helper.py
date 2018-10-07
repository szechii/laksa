'''
author:         szechi
date created:   2 Oct 2018
description:    Some miscellaneous helper functions
'''

import numpy as np
import itk
import scipy.ndimage
from PIL import Image
import nibabel as nib
import os

def makeOneCube(array, centroid, cube_shape = [40, 40, 40]):
    '''
    Extracts a cube from a 3D array.

    Input Args:     array           - 3D array from which cube will be extracted from (np.array)
                    centroid        - (z, y, x) coordinates where cube will be centred at (np.array)
                    cube_shape      - [z, y, x] shape of the desired cube (np.array)

    Output Args:    extracted_cube  - extracted cube from array of shape cube_shape (np.array)
                    coordinates     - [1, 6] array that indicates the z, y, x indices of \
                                    the input array where cube was extracted (np.array)
    '''
    assert len(cube_shape) == 3,        'Error: cube_shape is not length 3.'
    assert len(centroid.shape) == 3,    'Error: centroid is not length 3.'
    assert len(array.shape) == 3,       'Error: array is not 3D.'
    assert (cube_shape[0]%2 == 0 and \
            cube_shape[1]%2 == 0 and \
            cube_shape[2]%2 == 0),      'Error: cube sides are not even.'
    d, h, w = int(cube_shape[0]/2), int(cube_shape[1]/2), int(cube_shape[2]/2)
    z,y,x = centroid[0], centroid[1], centroid[2]
    minZ,minY,minX = z-d, y-h, x-w
    boundX = np.max((0,-minX))
    boundY = np.max((0,-minY))
    boundZ = np.max((0,-minZ))
    maxZ,maxY,maxX = z + d + boundZ, y + h + boundY, x + w + boundX
    minZ,minY,minX = np.max((0,minZ)), np.max((0,minY)), np.max((0,minX))
    boundZ = np.max((0,maxZ-array.shape[0]))
    boundY = np.max((0,maxY-array.shape[1]))
    boundX = np.max((0,maxX-array.shape[2]))
    minZ,minY,minX = minZ - boundZ, minY - boundY, minX - boundX
    coordinates = np.array([minZ, maxZ, minY, maxY, minX, maxX])
    extracted_cube = array[minZ:maxZ,minY:maxY,minX:maxX]

    return extracted_cube, coordinates

def getHeadMask(array):
    '''
    Returns an approximate binary head mask given an inverse radon DSA head reconstruction.
    Note that the first 10 z-slices of the input array will be ignored.
    This is because the first few z-slices of the inverse radon output are often empty, \
    hence connected threshold hole-filling on them will not be meaningful. 10 is an approximation.

    Input Args:     array       - 3D inverse radon output of the head (np.array)

    Output Args:    head_mask   - 3D binary head mask with dilation, iterations = 1 (np.array)
    '''
    assert len(array.shape) == 3, 'Error: array is not 3D.'

    head_mask = np.zeros(array.shape)
    for i in range(10,array.shape[0]):
        target = np.array(array[i,:,:]>0, np.float32)
        itk_im = itk.GetImageFromArray(target)
        imagetype = itk.Image[itk.F, 2]
        con_filt = itk.ConnectedThresholdImageFilter[imagetype, imagetype].New()
        con_filt.SetInput(itk_im)
        con_filt.SetUpper(0.1)
        con_filt.SetLower(0.0)
        con_filt.SetReplaceValue(255)
        con_filt.SetSeed([160, 160])
        con_filt.Update()
        head_mask[i,:,:] += scipy.ndimage.morphology.binary_closing(itk.GetArrayFromImage(con_filt.GetOutput()), iterations=5)
    head_mask = scipy.ndimage.morphology.binary_dilation(np.array(head_mask, np.int32))

    return head_mask

def makeCanvas(num_col, num_row, space, width, height):
    '''
    Makes a 2D RGBA pil image canvas large enough to paste height * width images \
    on a grid of num_col * num_row, with spacing between the images

    Input Args:     num_col - number of columns in the grid (int)
                    num_row - number of rows in the grid (int)
                    space   - amount of pixel space to be added between images (int)
                    width   - width of image (int)
                    height  - height of image (int)

    Output Args:    canvas  - pil image object (PIL Image)
    '''
    canvas = Image.fromarray(np.array(255*np.ones([num_row*(width+space), \
            num_col*(height+space)]),np.int32)).convert('RGBA')

    return canvas

def pil_img(img_arr, mask):
    '''
    Makes a 2D RGBA pil image with a red mask overlay.

    Input Args:     img_arr - 2D array (np.array)
                    mask    - 2D mask (np.array)

    Output Args:    final2  - pil image object (PIL Image)
    '''
    assert img_arr.shape ==  mask.shape,    'Error: img_arr and mask are not the same shape.'
    assert len(img_arr.shape) == 2,         'Error: img_arr is not 2D.'
    assert len(mask.shape) == 2,            'Error: mask is not 2D.'

    layer1 = Image.fromarray(np.uint8(255*(img_arr/np.max(img_arr))))
    mask = np.stack((mask,np.zeros(mask.shape),np.zeros(mask.shape)), axis=-1)
    layer2 = Image.fromarray(np.uint8(mask*255))
    final2 = Image.new("RGBA", layer1.size)
    layer2.putalpha(50)
    final2 = Image.alpha_composite(final2, layer1.convert('RGBA'))
    final2 = Image.alpha_composite(final2, layer2.convert('RGBA'))

    return final2

def listCentroids(array, cube_shape, stride):
    '''
    Returns a list of centroids of cubes when the 3D array is diced up.
    If (stride < side of cube), then cubes will have overlap of (side of cube - stride).

    Input Args:     array       - 3D array to dice up (np.array)
                    cube_shape  - shape of the cube (np.array)
                    stride      - stride size for dicing (int)

    Output Args:    mesh        - list of [z, y, x] indices (np.array)
    '''
    assert len(array.shape) == 3,       'Error: array is not 3D.'
    assert len(cube_shape) == 3,        'Error: cube_shape is not length 3.'
    assert type(stride) == int,         'Error: stride is not int.'
    assert (cube_shape[0]%2 == 0 and \
            cube_shape[1]%2 == 0 and \
            cube_shape[2]%2 == 0),      'Error: cube is not square.'
    assert stride <= cube_shape[0],     'Error: stride is not <= each side of the cube.'

    in_shape = array.shape
    d, h, w = int(cube_shape[0]/2), int(cube_shape[1]/2), int(cube_shape[2]/2)
    z_minbound, z_maxbound = d, in_shape[0]-d
    y_minbound, y_maxbound = h, in_shape[1]-d
    x_minbound, x_maxbound = w, in_shape[2]-d
    range_z = np.arange(z_minbound, z_maxbound+stride, stride)
    range_y = np.arange(y_minbound, y_maxbound+stride, stride)
    range_x = np.arange(x_minbound, x_maxbound+stride, stride)
    mesh = np.array(np.meshgrid(range_z, range_y, range_x)).T.reshape(-1,3)
    return mesh

def returnBox(mask):
    '''
    Returns the range of x, y, z indices that bounds a binary mask.

    Input Args:     mask                - binary mask (np.array)

    Output Args:    x_min, x_max, \
                    y_min, y_max, \
                    z_min, z_max        - range of x, y, z indices (list of ints)
    '''
    assert len(mask.shape) == 3, 'Error: Mask is not 3D.'
    x_list, y_list, z_list = np.where(mask>0)
    x_min = np.min(x_list)
    x_max = np.max(x_list)
    y_min = np.min(y_list)
    y_max = np.max(y_list)
    z_min = np.min(z_list)
    z_max = np.max(z_list)

    return x_min, x_max, y_min, y_max, z_min, z_max
