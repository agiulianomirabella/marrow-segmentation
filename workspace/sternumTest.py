import numpy as np

from rootPackages.segmentation.sternum.sternumCCL import extractSternumBoneMarrow_ccl
from rootPackages.utils.dataExtraction import readData
from rootPackages.utils.dataExploration import exploreVolumeOfInterest, displayImage, printImageInfo

'''
This module is intended to test the effectiveness of sternum bone marrow segmentation modules algorithms
'''

image = readData()

if __name__ == "__main__":
    
    #reducedSliceParameters = (zLimitsParameters, xLimitsParameters, yLimitsParameters, centerParameters)
    reducedSliceParameters = ([0.2, 0.8], [0.4, 0.6], [0.8, 1], [0.45, 0.55])

    printImageInfo(image)
    sternumBoneMarrow = extractSternumBoneMarrow_ccl(image, reducedSliceParameters)
    exploreVolumeOfInterest(image, sternumBoneMarrow)

