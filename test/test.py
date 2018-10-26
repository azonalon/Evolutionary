import matplotlib.pyplot as plt
import numpy as np

m = np.loadtxt("build/test.tiff")
plt.imshow(m, cmap='flag', origin='lower')
