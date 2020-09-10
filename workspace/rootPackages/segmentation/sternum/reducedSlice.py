import numpy as np

from rootPackages.utils.dataPreProcessing import digitizeToEqualWidth, binarize, centerImage
from scipy.ndimage import binary_opening
from scipy.ndimage.morphology import generate_binary_structure
from skimage.feature import canny

def extractReducedSlice(s, xLimitsParameters, yLimitsParameters, centerParameters):
    xLimits = [int(s.shape[0]*xLimitsParameters[0]), int(s.shape[0]*xLimitsParameters[1])]
    yLimits = [int(s.shape[1]*yLimitsParameters[0]), int(s.shape[1]*yLimitsParameters[1])]
    xWidth = xLimits[1] - xLimits[0]
    xCenter = [int(xWidth*centerParameters[0]), int(xWidth*centerParameters[1])]
    return xLimits, yLimits, xWidth, xCenter

def reducedSlicePreprocessing(reducedSlice):
    reducedSlice = digitizeToEqualWidth(reducedSlice, 10)
    reducedSlice = binarize(reducedSlice, np.unique(reducedSlice)[-1]) #binarize to the brightest gray
    reducedSlice = binarize(reducedSlice, 0) #change 0 to 1 and viceversa
    st = generate_binary_structure(2, 2) #generate connectivity pattern
    reducedSlice = binary_opening(reducedSlice, st) #compute binary opening to create CC
    reducedSlice = reducedSlice.astype(int) #because binary opening returns a bool array
    return reducedSlice

'''
    reducedSlice = setPercentileGrayLimits(reducedSlice, 20, 70)
    reducedSlice = digitizeToEqualWidth(reducedSlice, 3)
    reducedSlice = binarize(reducedSlice, np.unique(reducedSlice)[-1])
    reducedSlice = binarize(reducedSlice, 0)
    reducedSlice = canny(reducedSlice, sigma=10)
    return reducedSlice
'''
