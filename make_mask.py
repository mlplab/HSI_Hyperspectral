# coding: utf-8


import os
import scipy.io
import numpy as np
np.random.seed(1)


data_name = 'CAVE'
mask_size = {'CAVE': (512, 512, 31), 'Harvard': (1040, 1392, 31), 'ICVL': (1300, 1392, 31)}
save_path = f'../SCI_dataset/My_{data_name}'


mask = np.random.choice((0, 1), size=mask_size[data_name], p=(.5, .5))
scipy.io.savemat(os.path.join(save_path, 'test_mask.mat'), {'data': mask})