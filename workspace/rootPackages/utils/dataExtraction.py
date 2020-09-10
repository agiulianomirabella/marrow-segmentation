import random as rnd
import numpy as np
import skimage.io as io
from rootPackages.utils.dataPreProcessing import normalize, digitizeToEqualWidth
import os
import pydicom as dcm

def readImage(path):
    #given an image path, return the image as an numpy array
    return normalize(np.array(io.imread(path)))

def readDicomImageAsArray(path):
    #given a dicom file folder path, return its image as a numpy array
    if path[-1] != "/":
        path = path + "/"
    ds = []
    for i in os.listdir(path):
        ds.append(dcm.dcmread(path + i))
    slices = []
    for f in ds:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)

    # ensure they are in the correct order
    image = np.array([s.pixel_array for s in sorted(slices, key=lambda s: s.SliceLocation)])
    return normalize(np.swapaxes(image[::-1, ::-1, :], 1, 2))


def readData():
    path = "/home/giuliano/Desktop/tfg/workspace/rootPackages/data/ESTRUCTURAS/"
    return readDicomImageAsArray(path)
    #return digitizeToEqualWidth(readDicomImageAsArray(path), 256)

'''
def practiceImages(parameter, random= False):
    #if parameter is an integer then:
        #return a list of n images from practice_image database. Cases wil be chosen by their order in the directory.
        #If random then they will be chosen randomly
    #else if parameter is a list:
        #return a list of images whose indices match those in parameter

    out = []
    #LUNG1-007, -036, -050, -058, -067 are not available
    forbidden = [6, 35, 49, 57, 66]
    auxPath = "C:/Users/A. Giuliano/Desktop/tfg/VSCodeWS_TFG/practice_images/NSCLC-Radiomics/"
    studies = os.listdir(auxPath)

    selected = []

    if isinstance(parameter, int):
        while len(selected) != parameter:
            if random:
                selected = rnd.sample(range(len(studies)), parameter)
            else:
                selected = [i for i in range(parameter)]
            
            for e in selected:
                if e in forbidden:
                    selected = []

    elif isinstance(parameter, list):
        selected = [e - 1 for e in parameter]
        for e in selected:
            if e in forbidden:
                print("ERROR: the given indices list contains forbidden images. Forbidden are [7, 36, 50, 58, 67]")
                return None
    else:
        print("ERROR: parameter should be an integer or a list")
        return None

    for i in selected:
        possibleSecondPath = os.listdir(auxPath + studies[i])

        for p in possibleSecondPath:
            if "Study" in p:
                secondPath = p
                thirdPath = os.listdir(auxPath + studies[i] + "/" + secondPath)[0]
                completePath = auxPath + studies[i] + "/" + secondPath + "/" + thirdPath
                out.append(readDicomImageAsArray(completePath))

    print()
    print("CHOSEN STUDIES:")
    for i in selected:
        print(studies[i])
    
    return out
'''
