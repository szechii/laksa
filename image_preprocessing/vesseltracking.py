'''
author:         szechi
date created:   7 Oct 2018
description:    function to do vessel tracking through post-processing \
                vessel centrelines from VMTK vmtknetworkextraction.
                the function links up broken vessel centrelines using ball-tree \
                nearest neighbour and converts vessel centrelines into a \
                graph network with nodes and edges.
                users can rely on graph output to conduct tree searching \
                algorithms e.g. depth-first-search/ breadth-first-search.

about vmtk:     to obtain centrelines .vtp file:

                vmtkimagereader -ifile <original_image_file>
                --pipe vmtklevelsetsegmentation -ifile <vessel_segmentation_file>
                --pipe vmtkmarchingcubes -i @.o -ofile <centrelines_file_path>

                to obtain centrelines info .txt file:
                vmtkimagereader -ifile <vessel_segmentation_file> >centrelines_info_txtfile &
'''

import sys
import numpy as np
import pandas as pd
import vtk
import networkx as nx
from vtk.numpy_interface import dataset_adapter as dsa
from sklearn.neighbors import NearestNeighbors

def CentrelinesInfo(centrelines_file_path, centrelines_info_txtfile):
    '''
    Vessel centrelines post-processing and graph network

    Input Args:     centrelines_file_path       - .vtp file directory of vessel centrelines (string)
                    centrelines_info_txtfile    - .txt text file containing vmtkimagereader \
                                                output of the centrelines .vtp file (string)

    Output Args:    G                           - graph network (networkx object)
                    radius                      - list of vessel radii at each graph node. \
                                                List is arranged by graph node indices (list)
                    dicomidx                    - list of [x, y, z] physical coordinates of each graph node. \
                                                List is arranged by graph node indices (list)
    '''
    #read centrelines .vtp file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(centrelines_file_path)
    reader.Update()
    surface = reader.GetOutput()

    #create a dictionary of centrelines xyz points, radiuses, and connectivity information
    surfwrapper = dsa.WrapDataObject(surface)
    ArrayDict = dict()
    ArrayDict['Points'] = np.array(surfwrapper.Points)
    ArrayDict['Radius'] = np.array(surfwrapper.PointData.GetArray(surfwrapper.PointData.keys()[0]))
    cellPointIdsList = []
    numberOfCells = surfwrapper.VTKObject.GetNumberOfCells()
    for cellId in range(numberOfCells):
        cell = surfwrapper.VTKObject.GetCell(cellId)
        numberOfPointsPerCell = cell.GetNumberOfPoints()
        cellArray = np.zeros(shape=numberOfPointsPerCell, dtype=np.int32)
        for point in range(numberOfPointsPerCell):
            cellArray[point] = cell.GetPointId(point)
        cellPointIdsList.append(cellArray)
    ArrayDict['Connectivity'] = cellPointIdsList
    links = ArrayDict['Connectivity'] #contains indices of connected points grouped in arrays
    radius = ArrayDict['Radius']

    #convert xyz cartesian coordinates to ijk physical coordinates
    points = ArrayDict['Points'] #xyz coordinates of centerline points
    file_object  = open(centrelines_info_txtfile, "r")
    strings = file_object.read().split('\n')
    xyzras = strings[len(strings)-2]
    rasijk = strings[len(strings)-3]
    xyztoras_mat = np.array([float(x) for x in xyzras.split('[')[1].split(']')[0].split(',')]).reshape(-1,4)
    rastoijk_mat = np.array([float(x) for x in rasijk.split('[')[1].split(']')[0].split(',')]).reshape(-1,4)
    x = pd.DataFrame(points)[0]
    y = pd.DataFrame(points)[1]
    z = pd.DataFrame(points)[2]
    d = {'col1':np.array(x), 'col2': np.array(y), 'col3':np.array(z), 'col4':np.ones(x.shape[0])}
    coordinates = np.array(pd.DataFrame(data=d))
    rasidx = [np.dot(xyztoras_mat,i) for i in coordinates]
    dicomidx = [np.dot(rastoijk_mat,i) for i in rasidx]
    dicomidx = np.round(dicomidx)
    dicomidx = dicomidx[:,0:3] #ijk physical coordinates

    #use ball-tree nearest-neighbour to connect broken lines, with ball radius R
    nn_pool = np.zeros((len(links)*2,3))
    nn_links = np.zeros(len(links)*2, dtype=int)
    nn_dicomidx = np.zeros(len(links)*2, dtype=int)
    for i in range(len(links)):
        length = len(links[i])
        nn_pool[2*i] = dicomidx[links[i][0]]
        nn_pool[(2*i)+1] = dicomidx[links[i][length-1]]
        nn_dicomidx[2*i] = links[i][0]
        nn_dicomidx[(2*i)+1] = links[i][length-1]
        nn_links[2*i] = i
        nn_links[(2*i)+1] = i

    R = 0.1
    nbrs = NearestNeighbors(radius=R).fit(nn_pool)
    edgeDict = dict()
    for n in range(len(links)):
        start_idx = nn_dicomidx[2*n]
        end_idx = nn_dicomidx[(2*n)+1]
        start_coord = nn_pool[2*n]
        end_coord = nn_pool[(2*n)+1]
        startnb_distances, startnb_indices = nbrs.radius_neighbors([start_coord])
        endnb_distances, endnb_indices = nbrs.radius_neighbors([end_coord])

        startnb_indices_remove_source = np.delete(startnb_indices[0],np.where(startnb_indices[0]==2*n))
        endnb_indices_remove_source = np.delete(endnb_indices[0],np.where(endnb_indices[0]==(2*n)+1))

        if sum(startnb_indices_remove_source==(2*n)+1)!=1:
            startnb_indices_remove_source = np.append(startnb_indices_remove_source, [(2*n)+1])
        if sum(endnb_indices_remove_source==(2*n))!=1:
            endnb_indices_remove_source = np.append(endnb_indices_remove_source, [(2*n)])

        edgeDict[int(nn_dicomidx[2*n])] = [nn_dicomidx[x].astype(int) for x in startnb_indices_remove_source]
        edgeDict[int(nn_dicomidx[(2*n)+1])] = [nn_dicomidx[x].astype(int) for x in endnb_indices_remove_source]

    #draw graph G
    edges = [(i,x) for i in edgeDict for x in edgeDict[i]]
    edges_original = []
    for l in links:
        for index in range(len(l)-1):
            edges_original.append((l[index], l[index+1]))
    edges_final = set(edges_original+edges)
    G= nx.Graph()
    G.add_nodes_from(np.concatenate(links))
    G.add_edges_from(edges_final)

    return G, radius, dicomidx
