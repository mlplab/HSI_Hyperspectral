# coding: UTF-8


import os
import h5py as h5
import numpy as np
import scipy.io
from tqdm import tqdm


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


data_dir = '../SCI_dataset'
img_dir = os.path.join(data_dir, 'ICVL_2020_h5_before')
img_list = os.listdir(img_dir)
img_list.sort()
save_dir = os.path.join(data_dir, 'ICVL_2020_h5')
os.makedirs(save_dir, exist_ok=True)


with tqdm(img_list, ascii=True) as pbar:
    for i, img_name in enumerate(pbar):
        pbar.set_postfix({'name': f'{img_name:>30}'})
        if os.path.exists(os.path.join(save_dir, img_name)):
            continue
        try:
            open_format = 'h5'
            with h5.File(os.path.join(img_dir, img_name), 'r') as f:
                data = np.array(f['rad'])
                data = normalize(data.transpose((1, 2, 0))[::-1])
        except OSError:
            f = scipy.io.loadmat(os.path.join(img_dir, img_name))
            data = normalize(np.array(f['ref']))

        with h5.File(os.path.join(save_dir, img_name), 'w') as f:
            f.create_dataset('data', data=data)
