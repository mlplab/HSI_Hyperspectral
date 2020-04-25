# coding: utf-8


import os
import sys
import h5py
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


print('load icvl')
icvl_f = h5py.File('icvl.mat', 'r')
icvl_data = icvl_f['rad']
print('harvard')
harvard_f = scipy.io.loadmat('harvard.mat')
harvard_data = harvard_f['ref']
print('load dataset')
dataset_f = scipy.io.loadmat('dataset/train/0.mat')
dataset_data = dataset_f['data']
print(icvl_data.shape, harvard_data.shape, dataset_data.shape)
print(icvl_data.dtype, harvard_data.dtype, dataset_data.dtype)
print(type(icvl_data), type(harvard_data), type(dataset_data))
print(sys.getsizeof(icvl_data), sys.getsizeof(harvard_data), sys.getsizeof(dataset_data))
x = np.array(icvl_data)
print(x.nbytes, harvard_data.nbytes, dataset_data.nbytes)
print(icvl_f.keys(), harvard_f.keys(), dataset_f.keys())