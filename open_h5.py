# coding: UTF-8


import os
import h5py as h5
import scipy.io


data_dir = '../SCI_dataset'
img_dir = os.path.join(data_dir, 'ICVL_2020_h5_before')
img_list = os.listdir(img_dir)
img_list.sort()


for img_name in img_list:

    try:
        with h5.File(os.path.join(img_dir, img_name), 'r') as f:
            pass

    except OSError:
        print(img_name, 'continue')
        x = scipy.io.loadmat(os.path.join(img_dir, img_name))
        print(x.keys())
