# coding: UTF-8


import numpy as np
import matplotlib.pyplot as plt


x = np.random.random((512, 512, 3))
y = np.random.random((512, 512, 3))
z = np.abs(x - y)


fig, axes = plt.subplots(figsize=(16, 9), ncols=1, nrows=3)
im = axes[0, 0].imshow(x)
im = axes[0, 1].imshow(y)
im = axes[0, 2].imshow(z, cmap='jet')
cbar_ax = fig.add_axes(.87, axpos)