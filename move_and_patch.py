# coding: utf-8


import os
import shutil
from utils import make_patch


cave_path = '../CAVE/'
train_data_path = '../train_data'
test_data_path = '../test_data'
train_patch_path = '../train_patch_data'
test_patch_path = '../test_patch_data'


def move_data(data_path, data_list, move_path):
    os.makedirs(move_path, exist_ok=True)
    for name in tqdm(data_list):
        shutil.copy(os.path.join(data_path, name), os.path.join(move_path, name))
    return None


cave_list = os.listdir(cave_path)
train_list = cave_list[:20]
test_list = cave_list[20:]
print(len(train_list), len(test_list))
move_data(cave_path, train_list, train_data_path)
move_data(cave_path, test_list, test_data_path)
make_patch(train_data_path, train_patch_path, size=64, ch=31)
make_patch(test_data_path, test_patch_path, size=64, ch=31)