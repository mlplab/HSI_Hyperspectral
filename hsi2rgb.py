# coding: utf-8


import os
import shutil
import scipy.io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import normalize, plot_img


data_path = 'SCI_dataset'
result_path = 'result'
filter_path = os.path.join(data_path, 'D700_CSF.mat')
label_path = os.path.join(data_path, 'My_Harvard', 'eval_data')
img_path = os.path.join(result_path, 'output_mat')
output_path = os.path.join(result_path, 'rgb_img')
if os.path.exists(output_path):
    shutil.rmtree(otuput_path)
os.makedirs(output_path, exist_ok=True)


if __name__ == '__main__':

    filter_data = scipy.io.loadmat(filter_path)['T']
    label_list = os.listdir(label_path)
    img_list = os.listdir(img_path)
    img_list.sort()
    for i, img_name in enumerate(tqdm(img_list), ascii=True):
        plt.figure()
        img_f = scipy.io.loadmat(os.path.join(img_path, img_name))
        img_data = img_f['data']
        label_img = img_f['idx'][0]
        show_data = normalize(img_data.dot(filter_data)[:, :, ::-1])
        plt.subplot(121)
        plot_img(show_data, 'Output')
        label_f = scipy.io.loadmat(os.path.join(label_path, label_name))['data']
        show_data = normalize(label_data.dot(filter_data)[:, :, ::-1])
        plt.subplot(121)
        plot_img(show_data, 'Label')
        plt.imshow(show_data)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'output_{i:05d}_{label_idx}.png'), bbox_inches='tight')
        plt.close()