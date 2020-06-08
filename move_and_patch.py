# coding: utf-8


import os
import shutil
import numpy as np
from tqdm import tqdm
from utils import make_patch, patch_mask


data_name = 'CAVE'
data_path = f'../SCI_dataset/{data_name}'
save_path = f'../SCI_dataset/My_{data_name}'


train_data_path = os.path.join(save_path, 'train_data')
test_data_path = os.path.join(save_path, 'test_data')


train_patch_path = os.path.join(save_path, 'train_patch_data')
test_patch_path = os.path.join(save_path, 'test_patch_data')
eval_path = os.path.join(save_path, 'eval_data')
eval_show_path = os.path.join(save_show_path, 'eval_show_data')


mask_path = os.path.join(save_path, 'mask_data')
eval_mask_path = os.path.join(save_path, 'eval_mask_data')
mask_show_path = os.path.join(save_show_path, 'mask_show_data')


data_key = {'CAVE': 'im', 'Harvard': 'ref', 'ICVL': 'data'}
np.random.seed(1)


def move_data(data_path, data_list, move_path):
    os.makedirs(move_path, exist_ok=True)
    for name in tqdm(data_list):
        shutil.copy(os.path.join(data_path, name), os.path.join(move_path, name))
    return None


os.makedirs(save_path, exist_ok=True)
data_list = os.listdir(data_path)
data_list.sort()
data_list = np.array(data_list)
train_test_idx = np.random.choice((1, 2), data_list.shape[0], p=(.8, .2))
train_list = list(data_list[train_test_idx == 1])
test_list = list(data_list[train_test_idx == 2])
print(len(train_list), len(test_list))
move_data(data_path, train_list, train_data_path)
move_data(data_path, test_list, test_data_path)


make_patch(train_data_path, train_patch_path, size=64, step=64, ch=31, data_key=data_key[data_name])
make_patch(test_data_path, test_patch_path, size=64, step=64, ch=31, data_key=data_key[data_name])
make_patch(test_data_path, eval_path, size=256, step=256, ch=31, data_key=data_key[data_name])
make_patch(test_data_path, eval_show_path, size=512, step=512, ch=31, data_key=data_key[data_name])


patch_mask(os.path.join(data_path, 'test_mask.mat'), mask_path, size=64, step=64, ch=31)
patch_mask(os.path.join(data_path, 'test_mask.mat'), eval_mask_path, size=256, step=256, ch=31)
patch_mask(os.path.join(data_path, 'test_mask.mat'), mask_show_path, size=512, step=512, ch=31)