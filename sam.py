# coding: utf-8


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from model.unet_copy import UNet_none

# eps = 1e-7
#
# Lenna_bmp = 'Lenna.bmp'
# Lenna_100 = 'Lenna_100.jpg'
# Lenna_NN = 'Lenna_NN.png'
# mount = 'resize_mount.jpg'
#
# x = np.asarray(Image.open(Lenna_bmp), dtype=np.float32) / 255.
# y = np.asarray(Image.open(mount), dtype=np.float32) / 255.
# print(x.max(), x.min())
# print(y.max(), y.min())
# x_sqrt = np.sqrt(np.sum(x ** 2, axis=-1)) + eps
# y_sqrt = np.sqrt(np.sum(y ** 2, axis=-1)) + eps
# print(x_sqrt)
# print(y_sqrt)
#
#
# '''
# x = x.reshape(64 * 64, 3)
# x_sqrt = np.sqrt(np.sum(x ** 2, axis=1))
# y = y.reshape(64 * 64, 3)
# y_sqrt = np.sqrt(np.sum(y ** 2, axis=1))
# mul_x_y = x_sqrt * y_sqrt
#
# xy = np.sum(x * y, axis=1)
#
# angle = np.arccos(xy / mul_x_y)
# angle = angle.reshape((64, 64))
# print(angle.shape)
# '''
#
# xy = np.sum(x * y, axis=-1)
# print(x_sqrt * y_sqrt)
# print(xy)
# print(xy / (x_sqrt * y_sqrt))
#
# angle = np.arccos(xy / (x_sqrt * y_sqrt))
# show_angle = np.array(angle * 255., dtype=np.uint8)
# print(angle.shape)
# print(angle)
# print(np.cos(angle))
# plt.imshow(show_angle)
# plt.show()
#

model = UNet_none(25, 24)
summary(model, (25, 256, 256))
