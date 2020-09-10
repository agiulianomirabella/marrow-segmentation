from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

ellipse = Ellipse([100, 100], 100, 50, angle=0)

path = ellipse.get_path()
# Get the list of path vertices
vertices = path.vertices.copy()
# Transform the vertices so that they have the correct coordinates
vertices = ellipse.get_patch_transform().transform(vertices)

print(vertices.shape)

img = np.zeros((500, 500))

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap='gray')
ax.plot(vertices[:, 1], vertices[:, 0], '-b', lw=3)
plt.show()
