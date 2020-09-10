import numpy as np
from skimage.feature import canny

from rootPackages.utils.dataExtraction import readData
from rootPackages.utils.dataExploration import display3DImage, printImageInfo
from rootPackages.segmentation.sternum.sternumUtils.auxiliaryReducedSlice import extractAllReducedSlices, displayReducedSlices
from rootPackages.utils.dataPreProcessing import digitizeToEqualWidth

from rootPackages.utils.dataPreProcessing import setPercentileGrayThresholds, digitizeToEqualWidth, binarize

image = readData()
slices = extractAllReducedSlices(image, ([0.2, 0.8], [0.4, 0.6], [0.8, 1]))
'''
for i, s in enumerate(slices):
    s = setPercentileGrayThresholds(s, 20, 70)
    s = digitizeToEqualWidth(s, 3)
    s = binarize(s, np.unique(s)[-1])
    s = binarize(s, np.unique(s)[0])

    s = canny(s, sigma=2)
    slices[i] = s
'''
displayReducedSlices(slices)