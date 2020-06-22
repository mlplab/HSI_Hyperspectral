# coding: utf-8


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# from colour_func import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from colour.colorimetry import transformations
from utils import normalize


func_name = transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
start_wave = 400
last_wave = 700
# data_name = 'dataset/icvl/mat/icvl.mat'
data_name = 'dataset/train/balloons_ms.mat'
x = np.arange(start_wave, last_wave + 1, 10)


f = scipy.io.loadmat(data_name)
data = normalize(f['im'])
# rgb = normalize(f['rgb'])
# plt.imsave('output_img/label.png', rgb)
print(data.shape)
trans_filter = func_name(x)
for i, ch in enumerate(range(start_wave, last_wave + 1, 10)):
     break
     print(i, ch)
     # trans_data = normalize(np.expand_dims(data[:, :, 10], axis=-1).dot(np.expand_dims(trans_filter[10], axis=0)))
     trans_data = normalize(np.expand_dims(data[:, :, i], axis=-1).dot(np.expand_dims(trans_filter[i], axis=0)))
     # print(trans_data.shape)
     # show_data = trans_data
     # show_data = data[:, :, (21, 7, 2)]
     show_data = trans_data
     # plt.imshow(show_data)
     plt.imsave(f'output_img/{i}_{ch}.png', show_data)
     plt.close()
     # plt.show()
g_ch = 16
b_ch = 9
for i in range(21, 31):
     show_data = normalize(data[:, :, (i, g_ch, b_ch)])
     # show_data = trans_data
     # plt.imshow(show_data)
     plt.imsave(f'output_img/RGB_{i}_{g_ch}_{b_ch}.png', show_data)
     # plt.savefig(f'output_img/RGB_{i}_{g_ch}_{b_ch}.png')
     # plt.close()
     # plt.show()