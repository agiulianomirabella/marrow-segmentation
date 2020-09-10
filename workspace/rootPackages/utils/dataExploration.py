import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from multiprocessing import Process
from skimage.color import gray2rgb

'''
This module will provide some helpful auxiliary functions
'''

def printImageInfo(image):
    #print image shape, min and max
    print('\nIMAGE INFO:')
    print('Image shape: {}; min: {}, max: {}; number of different gray levels: {}\n'.format(np.shape(image),
        np.min(image), np.max(image), len(np.unique(image))))

def printTime(start, description = 'Execution'):
    #given a start time, print the difference between actual time and start
    print(description + ' time: {}'.format(round(time.time() - start, 2)))

def display2DImage(image):
    #display a 2D image
    image = np.swapaxes(image[:, ::-1], 0, 1)
    plt.imshow(image, cmap = 'gray')
    plt.show(block = True)

def display3DImage(image, view = 'transverse'):
    #display 3D images slices. Press j to show previous slice and k to show next one
    image = np.swapaxes(image[:, :, ::-1], 1, 2)
    if view == 'frontal':
        image = np.swapaxes(image, 0, 1)
    elif view == 'sagittal':
        image = np.swapaxes(image, 0, 2)
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = image
    ax.index = image.shape[0] // 2
    ax.imshow(image[ax.index], cmap = 'gray')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show(block = True)

def displayImage(image, view = 'transverse'):
    if image.ndim == 2:
        display2DImage(image)
    elif image.ndim == 3:
        display3DImage(image, view)
    else:
        print("ERROR: image is neither 3- nor 2-dimensional")

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
        time.sleep(0.1)
    elif event.key == 'k':
        next_slice(ax)
        time.sleep(0.1)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def displayHistogram(image, axis= [], omitZeroValues= True):
    f = image.flatten()
    if omitZeroValues:
        f = f[f!=0]
    plt.hist(f,  bins= 'auto')
    if axis:
        plt.axis(axis)
    plt.ylabel('Frequency')
    plt.xlabel('Gray Values')
    plt.xlim(left = 0)
    if omitZeroValues:
        plt.title('Image Histogram (zero values omitted).')
    else:
        plt.title('Image Histogram.')
    plt.show(block=True)

def exploreImage(image, omitZeroValuesInHistogram= True):
    #print the image info, then display the image and its histogram
    procs = []

    proc1 = Process(target=printImageInfo, args=(image,))
    procs.append(proc1)
    proc2 = Process(target=displayImage, args=(image,))
    procs.append(proc2)
    proc3 = Process(target=displayHistogram, args=(image, omitZeroValuesInHistogram))
    procs.append(proc3)
    proc1.start()
    proc2.start()
    proc3.start()

def exploreVolumeOfInterest(image, volumeCoordinates):
    print("Volume size: {};".format(len(volumeCoordinates)))
    aux = gray2rgb(image)

    for c in volumeCoordinates:
        aux[tuple(c)] = [0, 0, 1]

    #display 3D images slices. Press j to show previous slice and k to show next one
    aux = np.swapaxes(aux[:, :, ::-1], 1, 2)
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = aux
    ax.index = aux.shape[0] // 2
    ax.imshow(aux[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show(block = True)

