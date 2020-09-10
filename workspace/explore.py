from rootPackages.utils.dataExploration import exploreImage, displayImage
from rootPackages.utils.dataExtraction import readData
from rootPackages.utils.dataPreProcessing import digitizeToEqualWidth, digitizeToEqualFrequencies
from skimage.filters import threshold_multiotsu

'''
Quick visual exploration
'''

image = readData()
image = digitizeToEqualFrequencies(image, 256)
aux = digitizeToEqualWidth(image, 256)

'''
t = threshold_multiotsu(image, 5)
background = t[1]
image[image<background]=0
'''

exploreImage(image)
