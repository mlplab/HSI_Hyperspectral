# coding: UTF-8


import os
import h5py
import numpy as np
from tqdm import tqdm


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


data_dir = '../SCI_dataset'
img_dir = os.path.join(data_dir, 'ICVL_2020_h5_before')
img_list = os.listdir(img_dir)
img_list.sort()
save_dir = os.path.join(data_dir, 'ICVL_2020_h5')
os.makedirs(save_dir, exist_ok=True)


for img_name in tqdm(img_list):
    with h5py.File(os.path.join(img_dir, img_name), 'r') as f:
        data = np.array(f['rad'])
        data = normalize(data.transpose((1, 2, 0))[::-1])
        bands = np.array(f['bands'])
        rgb = np.array(f['rgb'])

    with h5py.File(os.path.join(save_dir, img_name), 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('bands', data=bands)
        f.create_dataset('rgb', data=rgb)
