# coding: UTF-8


import os
import h5py


data_dir = '../SCI_dataset/ICVL_2020_h5'


if __name__ == '__main__':

    data_list = os.listdir(data_dir)
    data_list.sort()
    for name in data_list:
        try:
            f = h5py.File(os.path.join(data_dir, name), 'r')
            print(f'{name:50s}')
        except OSError:
            print(f'{name:50s}, except')
