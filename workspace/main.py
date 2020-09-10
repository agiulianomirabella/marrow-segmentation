from rootPackages.segmentation.sternum.reducedSlice import extractReducedSlice
from rootPackages.utils.dataExtraction import readData
from rootPackages.utils.dataExploration import displayImage, displayHistogram, printImageInfo
from rootPackages.utils.dataPreProcessing import centerImage

reducedSliceParameters = ([0.2, 0.8], [0.4, 0.6], [0.8, 1], [0.45, 0.55])

image = readData()
image[image<0.4] = 0
s = centerImage(image[45])[0]
printImageInfo(s)
s = s[180:260, 250:]

