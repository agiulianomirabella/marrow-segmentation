import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_multiotsu
from skimage.feature import canny
from skimage.segmentation import active_contour

from rootPackages.utils.dataExtraction import readImage
from rootPackages.utils.dataPreProcessing import digitizeToEqualWidth, setPercentileGrayThresholds, setGrayThresholds, binarize

img = readImage('test.png')[100:380, 100:540]
img = rgb2gray(img)

img = setPercentileGrayThresholds(img, 20, 70)
img = digitizeToEqualWidth(img, 3)
img = binarize(img, np.unique(img)[-1])
img = binarize(img, np.unique(img)[0])

img = canny(img, sigma=1)

#create the ellipse:
ellipse = Ellipse([130, 200], 90, 280)
path = ellipse.get_path()
# Get the list of path vertices:
vertices = path.vertices.copy()
# Transform the vertices so that they have the correct coordinates:
vertices = ellipse.get_patch_transform().transform(vertices)
init = vertices

snake = active_contour(img, init, alpha=0.025, beta=0.1,
                   w_line=0, w_edge=2, gamma=0.1,
                   bc=None, max_px_move=1.0,
                   max_iterations=2500, convergence=0.1,
                   boundary_condition='periodic',
                   coordinates='rc')

'''
def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   bc=None, max_px_move=1.0,
                   max_iterations=2500, convergence=0.1,
                   *,
                   boundary_condition='periodic',
                   coordinates=None):

    alpha=0.01:
        Snake length shape parameter. Higher values makes snake contract
        faster.
    beta=0.1:
        Snake smoothness shape parameter. Higher values makes snake smoother.

    w_line=0: float, optional
        Controls attraction to brightness. Use negative values to attract toward
        dark regions.
    w_edge=1: float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.

    gamma=0.01 : float, optional
        Explicit time stepping parameter.
    bc : deprecated; use ``boundary_condition``
        DEPRECATED. See ``boundary_condition`` below.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_iterations : int, optional
        Maximum iterations to optimize snake shape.
    convergence: float, optional
        Convergence criteria.
    boundary_condition : string, optional
        Boundary conditions for the contour. Can be one of 'periodic',
        'free', 'fixed', 'free-fixed', or 'fixed-free'. 'periodic' attaches
        the two ends of the snake, 'fixed' holds the end-points in place,
        and 'free' allows free movement of the ends. 'fixed' and 'free' can
        be combined by parsing 'fixed-free', 'free-fixed'. Parsing
        'fixed-fixed' or 'free-free' yields same behaviour as 'fixed' and
        'free', respectively.

'''

fig, ax = plt.subplots(figsize=(14, 14))
ax.imshow(img, cmap='gray')
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()
