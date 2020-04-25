# coding: utf-8


import os
import time
import scipy
import scipy.io
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import HyperSpectralDataset
from utils import RandomCrop
import torchvision


import torch


data_path = 'dataset/'


def make_patch(data_path, save_path):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    for i, name in enumerate(data_list):
        print(name)
        idx = name.split('.')[0]
        f = scipy.io.loadmat(os.path.join(data_path, name))
        data = f['data']
        data = np.expand_dims(np.asarray(
            data, np.float32).transpose([2, 0, 1]), axis=0)
        tensor_data = torch.as_tensor(data)
        patch_data = tensor_data.unfold(2, 256, 256).unfold(3, 256, 256)
        patch_data = patch_data.permute(
            (0, 2, 3, 1, 4, 5)).reshape(-1, 24, 256, 256)
        for i in range(patch_data.size()[0]):
            print(i)
            save_data = patch_data[i].to('cpu').detach().numpy().copy()
            save_name = os.path.join(save_path, f'{idx}_{i}.mat')
            scipy.io.savemat(save_name, {'data': save_data})

    return None


def make_spectral_data(data_path, save_path):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    for i, name in enumerate(data_list):
        print(name)
        idx = name.split('.')[0]
        f = scipy.io.loadmat(os.path.join(data_path, name))
        data = f['data']
        h, w, ch = data.shape
        for i in range(ch):
            print(i)
            save_data = np.expand_dims(data[:, :, i].to('cpu').detach().numpy().copy(), axis=-1)
            save_name = os.path.join(save_path, f'{idx}_{i}ch.mat')
            scipy.io.savemat(save_name, {'data': save_data})

    return None


if __name__ == '__main__':

    dataset_path = '../dataset/'
    img_path = '0.mat'
    output_name = 'output_ch_data'
    input_path = os.path.join(dataset_path, img_path)
    output_path = os.path.join(dataset_path, output_name)
    data = scipy.io.loadmat(input_path)['data']
    print(data.shape)
    os.makedirs(output_path, exist_ok=True)
    h, w, ch = data.shape
    for i in range(ch):
        print(i)
        save_data = np.expand_dims(data[:, :, i], axis=-1)
        save_name = os.path.join(output_path, f'test_output_data_{i}ch.mat')
        scipy.io.savemat(save_name, {'data': save_data})
