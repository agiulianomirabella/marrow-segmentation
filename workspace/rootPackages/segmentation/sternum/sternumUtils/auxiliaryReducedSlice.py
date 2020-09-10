import numpy as np
from rootPackages.segmentation.sternum.reducedSlice import extractReducedSlice, reducedSlicePreprocessing
from rootPackages.utils.dataPreProcessing import setGrayThresholds, setPercentileGrayThresholds, centerImage, binarize
from skimage.filters import threshold_multiotsu

import matplotlib.pyplot as plt
from rootPackages.utils.dataExploration import remove_keymap_conflicts, process_key

'''
Return a list of reducedWindow images of an image
'''

def padSlice(vector, deltaX, deltaY):
    return np.pad(vector, (deltaX, deltaY), 'edge')

def displayReducedSlices(image):
    #display 3D images slices. Press j to show previous slice and k to show next one

    print('\nREDUCED SLICES INFO:')
    print('Length: {}; min: {}, max: {}; number of different gray levels: {}\n'.format(len(image), min([np.min(s) for s in image]), max([np.max(s) for s in image]), len(set([e for s in image for e in np.unique(s)]))))

    for i, s in enumerate(image):
        image[i] = np.swapaxes(s[:, ::-1], 0, 1)
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = image
    ax.index = image.shape[0] // 2
    ax.imshow(image[ax.index], cmap = 'gray')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show(block = True)


def obtainReducedWindowImages(image, reducedSliceParameters):
    #return a list of reducedWindow of an image

    t = threshold_multiotsu(image, 5)
    background = t[1]

    out = []
    zLimitsParameters, xLimitsParameters, yLimitsParameters = reducedSliceParameters

    #center the image in z axis, in order to acquire a proper reduced image
    zLimits = [int(image.shape[0]*zLimitsParameters[0]), int(image.shape[0]*zLimitsParameters[1])]
    aux = image[slice(zLimits[0], zLimits[1])]

    for i, s in enumerate(aux):
        s = setGrayThresholds(s, lowerThreshold= background)
        s = setPercentileGrayThresholds(s, upperThresholdPercentile= 95)
        s, offSet = centerImage(s)

        #extract a reduced slice, (taking advantage of the almost constant sternum localization)
        xLimits, yLimits, xWidth, xCenter = extractReducedSlice(s, xLimitsParameters, yLimitsParameters, [0.4, 0.6])
        reducedSlice = s[slice(xLimits[0], xLimits[1]), slice(yLimits[0], yLimits[1])]
        out.append(reducedSlice)

    xmin = out[0].shape[0]
    ymin = out[0].shape[1]
    for s in out:
        if s.shape[0] < xmin:
            xmin = s.shape[0]
        if s.shape[1] < ymin:
            ymin = s.shape[1]

    aux = []
    for s in out:
        aux.append(s)
        #aux.append(s[:xmin, :ymin])

    return np.array(aux)

def extractAllReducedSlices(image, reducedSliceParameters = ([0.2, 0.8], [0.4, 0.6], [0.8, 1])):
    return obtainReducedWindowImages(image, reducedSliceParameters)
