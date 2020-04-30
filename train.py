# coding: utf-8

import os
import torch
import torchvision
from trainer import Trainer
from model.HIPN import HIPN
from data_loader import HyperSpectralDataset
from utils import RandomCrop, RandomHorizontalFlip, ModelCheckPoint


batch_size = 1
epochs = 5
# data_len = batch_size * 10

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

img_path = 'dataset/'
train_path = os.path.join(img_path, 'train')
test_path = os.path.join(img_path, 'test')
# drive_path = '/content/drive/My Drive/auto_colorization/'

train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
# transform = (RandomCrop(crop), RandomHorizontalFlip(),
#              torchvision.transform.ToTensor())
train_transform = None
# test_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = HyperSpectralDataset(train_path, os.path.join(img_path, 'test_mask.mat'), tanh=False, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = HyperSpectralDataset(test_path, os.path.join(img_path, 'test_mask.mat'), tanh=False, transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
model = HIPN(1, 31, block_num=3, ratio=1, output_norm=None).to(device)
criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
ckpt_cb = ModelCheckPoint('ckpt', 'HIPN', mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, callbacks=[ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
