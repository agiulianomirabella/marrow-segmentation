import numpy as np 

def makeUnique(voxelsList):
    #delete duplicates in a list of voxels coordinates
    return [np.array(a) for a in set([tuple(e) for e in voxelsList])]

def getGrayValueCoordinates(image, grayValue):
    #return a list of all coordinates where image ncells takes the value grayValue
    out = []
    info = np.where(image == grayValue)
    for i in range(len(info[0])):
        a = []
        for dim in info:
            a.append(dim[i])
        out.append(np.array(a))
    return out


'''
def addZeroMargins(image):
    #add zero margins to ndarray
    padWidth = tuple()
    constantValues = tuple()
    for _ in range(image.ndim):
        padWidth = padWidth  + ((1, 1), )
        constantValues = constantValues  + ((0, 0), )
    return np.pad(image, padWidth, 'constant', constant_values=constantValues)
'''
