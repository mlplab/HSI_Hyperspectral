# coding: utf-8


import os
import scipy.io
import numpy as np
from tqdm import tqdm
from utils import normalize


data_path = '../SCI_dataset/ICVL'
save_path = '../SCI_dataset/ICVL_transpose'
os.makedirs(save_path, exist_ok=True)


img_list = os.listdir(data_path)
img_list.sort()
for img_name in tqdm(img_list, ascii=True):
    f = scipy.io.loadmat(os.path.join(data_path, img_name))
    data = f['data'].transpose(1, 0, 2)[::-1]
    rgb = f['rgb']
    bands = f['bands']
    scipy.io.savemat(os.path.join(save_path, img_name), {'data': data, 'rgb': rgb, 'bands': bands})