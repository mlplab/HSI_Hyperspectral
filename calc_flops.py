# coding: UTF-8

import os
import pickle
import shutil
import scipy.io
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchsummary import summary
from trainer import Trainer
from model.HIPN import HSI_Network_share
from model.HSCNN import HSCNN
from model.hyperreconnet import HyperReconNet
from model.Ghost_HSCNN import Ghost_HSCNN
from data_loader import HyperSpectralDataset, PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, ModelCheckPoint, make_patch, PlotStepLoss, patch_mask, normalize, Draw_Output, plot_features
from ptflops import get_model_complexity_info


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True




block_num = 9
modes = ['mix1', 'mix2']
activations = ['relu']
s = [i for i in range(2, 4 + 1)]
all_macs = []
all_params = []
model_names = []
for activation in activations:
    for ratio in s:
        for mode in modes:
            model_name = f'ghost_{mode}_ratio_{ratio:02d}_activation_{activation}'
            model_name = f'{model_name}_{block_num}'
            print(model_name)
            model = Ghost_HSCNN(1, 31, ratio=ratio, activation=activation, mode=mode)
            model.to(device) 
            # summary(model, input_size=(1, 48, 48))
            macs, params = get_model_complexity_info(model, (1, 512, 512), verbose=True, print_per_layer_stat=False)
            model_names.append(model_name)
            all_macs.append(macs)
            all_params.append(params)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            input = torch.randn(1, 1, 48, 48)
