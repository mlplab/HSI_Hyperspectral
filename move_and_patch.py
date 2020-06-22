# coding: utf-8


import os
import random
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
eval_show_path = os.path.join(save_path, 'eval_show_data')


callback_path = os.path.join(save_path, 'callback_path')


mask_path = os.path.join(save_path, 'mask_data')
eval_mask_path = os.path.join(save_path, 'eval_mask_data')
mask_show_path = os.path.join(save_path, 'mask_show_data')


data_key = {'CAVE': 'im', 'Harvard': 'ref', 'ICVL': 'data'}
data_size = {'CAVE': (512, 512, 31), 'Harvard': (1040, 1392, 31), 'ICVL': (1300, 1392, 31)}
seed_key = {'CAVE': 1, 'Harvard': 2, 'ICVL': 3}
train_test_idx = {'CAVE': np.array([1] * 20 + [2] * 12),
                  'Harvard': np.array([1] * 40 + [2] * 10),
                  'ICVL': np.random.choice((1, 2), data_list.shape[0], p=(.8, .2))}
np.random.seed(seed_key[data_name])


def move_data(data_path, data_list, move_path):
    os.makedirs(move_path, exist_ok=True)
    for name in tqdm(data_list, ascii=True):
        shutil.copy(os.path.join(data_path, name), os.path.join(move_path, name))
    return None


os.makedirs(save_path, exist_ok=True)
mask_data = np.random.choice((0., 1.), data_size[data_name], p=(.5, .5))
scipy.io.savemat(os.path.join(save_path, 'mask.mat'), {'data': mask_data})
data_list = os.listdir(data_path)
data_list.sort()
data_list = np.array(data_list)
train_list = list(data_list[train_test_idx[data_name] == 1])
test_list = list(data_list[train_test_idx[data_name] == 2])
print(len(train_list), len(test_list))
move_data(data_path, train_list, train_data_path)
move_data(data_path, test_list, test_data_path)


make_patch(train_data_path, train_patch_path, size=64, step=64, ch=31, data_key=data_key[data_name])
make_patch(test_data_path, test_patch_path, size=64, step=64, ch=31, data_key=data_key[data_name])
make_patch(test_data_path, eval_path, size=256, step=256, ch=31, data_key=data_key[data_name])
make_patch(test_data_path, eval_show_path, size=512, step=512, ch=31, data_key=data_key[data_name])


patch_mask(os.path.join(data_path, 'mask.mat'), mask_path, size=64, step=64, ch=31)
patch_mask(os.path.join(data_path, 'mask.mat'), eval_mask_path, size=256, step=256, ch=31)
patch_mask(os.path.join(data_path, 'mask.mat'), mask_show_path, size=512, step=512, ch=31)


callback_list = os.listdir(eval_show_path)
callback_list = random.sample(callback_list, int(len(callback_list) * .3))
os.makedirs(callback_path, exist_ok=True)
move_data(eval_show_path, callback_list, callback_path)