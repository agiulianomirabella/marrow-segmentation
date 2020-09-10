import numpy as np
import copy
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, binary_closing
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import label, find_objects

from rootPackages.utils.dataExploration import displayImage
from rootPackages.utils.voxels import getGrayValueCoordinates
from rootPackages.utils.dataPreProcessing import digitizeToEqualWidth, setPercentileGrayThresholds, setGrayThresholds, binarize, centerImage
from rootPackages.segmentation.sternum.reducedSlice import extractReducedSlice, reducedSlicePreprocessing
from skimage.filters import threshold_multiotsu

'''
This module is intended to provide an algorithm for sternum bone marrow segmentation.
It performs a simple ccl algorithm on a reduced slice to identify black void inside the bone structure.
'''

def extractSternumBoneMarrow_ccl(image, reducedSliceParameters):
    #return all voxels coordinates (numpy array) identified as boneMarrow
    zLimitsParameters, xLimitsParameters, yLimitsParameters, centerParameters = reducedSliceParameters

    t = threshold_multiotsu(image, 5)
    background = t[1]

    out = []

    #center the image, in order to acquire a proper reduced image later
    if zLimitsParameters:
        zLimits = [int(image.shape[0]*zLimitsParameters[0]), int(image.shape[0]*zLimitsParameters[1])]
        aux = image[slice(zLimits[0], zLimits[1])]
        #show start and finish slices, to check
        #displayImage(aux[0])
        #displayImage(aux[-1])

    #Coordinates extraction
    for i, s in enumerate(aux):
        s = setGrayThresholds(s, lowerThreshold= background)
        s = setPercentileGrayThresholds(s, upperThresholdPercentile= 95)
        s, offSet = centerImage(s)

        #extract a reduced slice, (taking advantage of the almost constant sternum localization)
        xLimits, yLimits, xWidth, xCenter = extractReducedSlice(s, xLimitsParameters, yLimitsParameters, centerParameters)
        reducedSlice = s[slice(xLimits[0], xLimits[1]), slice(yLimits[0], yLimits[1])]
        #z, x, y

        #show some reducedSilces on slices to check their correctness
        if i%20 == 0:
            img = copy.copy(s)
            img[slice(xLimits[0], xLimits[1]), slice(yLimits[0], yLimits[1])] = 2
            displayImage(img)

        #Slice preprocessing
        reducedSlice = reducedSlicePreprocessing(reducedSlice)

        #label the reducedSlice:
        labeled = label(reducedSlice, structure= generate_binary_structure(2, 2))[0]
        objects_slices = find_objects(labeled)

        xMin = xWidth #initialize the highest height of CC
        l = -1 #in case no object is found

        #for each object of the slice, check if it contains the reducedSlice center and it is not background:
        for j, object_slice in enumerate(objects_slices):
            if object_slice[0].start != 0 and object_slice[0].stop != xWidth: #check it is not background
                if object_slice[0].start < xCenter[1] and object_slice[0].stop > xCenter[0]: #check it is "near" the center
                    if object_slice[1].start < xMin: #than check it is the higest one
                        l = j+1
                        xMin = object_slice[1].start

        if l != -1: #if an object has been found, than compute its coordinates and sum the offset due to reducedSlice extraction
            out = out + [np.insert(c, 0, i) + 
                np.array([zLimits[0], xLimits[0]+ offSet[0].start, yLimits[0] + offSet[1].start]) 
                for c in getGrayValueCoordinates(labeled, l)]

    return out

