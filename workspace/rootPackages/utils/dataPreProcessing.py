import numpy as np
import copy
from scipy.ndimage import label, find_objects
 
'''
This module will define functions to preProcess an image, before further analysis get done
'''

def normalize(image):
    #return the normalized image
    image = image - np.min(image)
    return image/np.max(image)

def binarize(image, grayLevel):
    #set grayLevel voxels to 1 and different to 0
    out = np.zeros(image.shape)
    out[image == grayLevel] = 1
    return out

def applyThreshold(image, grayLevel):
    #set grayLevel or less voxels to 0 and more-than-grayLevel voxels to 1
    out = copy.copy(image)
    out[image <= grayLevel] = 0
    out[image > grayLevel] = 1
    return out

def setGrayThresholds(image, lowerThreshold = 0, upperThreshold = 1):
    #set lower than lowerThreshold grayValues to 0 and higher than upperThreshold to upperThreshold

    if lowerThreshold > upperThreshold:
        raise ValueError('Upper threshold must be grater or equal than lower')

    out = copy.copy(image)
    if lowerThreshold < 0:
        lowerThreshold = 0
    if upperThreshold > 1:
        upperThreshold = 1

    out[image <= lowerThreshold] = 0
    out[image > upperThreshold] = upperThreshold
    return normalize(out)

def setPercentileGrayThresholds(image, lowerThresholdPercentile = 0, upperThresholdPercentile = 100):
    #Set lowerThresholdPercentile grayLeves to 0 and upperThresholdPercentile to 1

    if lowerThresholdPercentile < 0:
        lowerThresholdPercentile = 0
    if upperThresholdPercentile > 100:
        upperThresholdPercentile = 100

    lowerThreshold = np.percentile(image, lowerThresholdPercentile)
    upperThreshold = np.percentile(image, upperThresholdPercentile)

    #print()
    #print('The {} (lower) percentile gray level is: {}'.format(lowerThresholdPercentile, round(lowerThreshold, 2)))
    #print('The {} (upper) percentile gray level is: {}'.format(upperThresholdPercentile, round(upperThreshold, 2)))

    return setGrayThresholds(image, lowerThreshold, upperThreshold)

#TODO: check if time complexity improvement should be done
def digitizeToEqualFrequencies(image, binsNumber):
    #equal frequency binning of the image
    out = np.zeros(image.shape)
    labels = [i for i in range(binsNumber)]
    elements = np.sort(image.flatten())
    elementsPerBin = image.size//binsNumber
    bin_edges = [elements[i*elementsPerBin] for i in labels]
    for i, e in enumerate(bin_edges):
        out[image > e] = labels[i]
    return normalize(out)
    
def digitizeToEqualWidth(image, GrayLevelsNumber):
    #equal width binning of the image
    if GrayLevelsNumber > len(np.unique(image)):
        GrayLevelsNumber = len(np.unique(image))
    return normalize(np.digitize(image, bins = np.linspace(0, np.max(image), GrayLevelsNumber)) - 1)

#TODO: improve time complexity if necessary
def centerImage(image):
    #eliminate empty borders of the image
    aux = copy.copy(image)
    aux[image>0] = 1
    aux = aux.astype(int)
    margins = find_objects(aux)[0]
    return image[margins], margins
