import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['toolbar'] = 'None'

X_SIZE = 21796
Y_SIZE = 13660
X_SPACING = 500
Y_SPACING = 500
X_EDGE = 148
Y_EDGE = 80

fig = plt.figure()
ax  = fig.add_axes([148/21796, 80/13660, 1-2*148/21796, 1-2*80/13660])

# Make a grid...
nrows, ncols = 27,43
image = np.zeros(nrows*ncols)

# Set every other cell to a random number (this would be your data)
image[::2] = 1

image = image.reshape((nrows, ncols))

ax.matshow(image)
plt.show(fig)