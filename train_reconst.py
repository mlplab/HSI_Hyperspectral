# coding: utf-8

import os
import shutil
import torch
import torchvision
# from trainer import Trainer
from gan_trainer import Deep_GAN_Trainer, GAN_Trainer
# from model.unet import UNet
from model.unet_copy import Deeper_UNet
from model.discriminator import Discriminator
from data_loader import HyperSpectralDataset
from utils import RandomCrop, RandomHorizontalFlip, ModelCheckPoint


crop = 256
batch_size = 1
epochs = 5
# data_len = batch_size * 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

ckpt_path = 'ckpt'
# shutil.rmtree(ckpt_path)
os.mkdir(ckpt_path)

g_ckpt_path = os.path.join(ckpt_path, 'deep_unet')
d_ckpt_path = os.path.join(ckpt_path, 'du_discriminator')
# drive_path = '/content/drive/My Drive/auto_colorization/'


img_path = 'dataset/'
train_path = os.path.join(img_path, 'train')
test_path = os.path.join(img_path, 'test')
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
transform = (RandomCrop(crop), RandomHorizontalFlip(),
             torchvision.transforms.ToTensor())
train_dataset = HyperSpectralDataset(
    train_path, os.path.join(img_path, 'mask.mat'), transform=transform)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print('load traindata')
test_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_dataset = HyperSpectralDataset(
    test_path, os.path.join(img_path, 'mask.mat'), transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print('load testdata')


g_model = Deeper_UNet(25, 24, hcr=True, attention=False).to(device)
print('load G model')
d_model = Discriminator([32, 64, 128], (crop, crop), 25).to(device)
print('load D model')
g_loss = [torch.nn.MSELoss().to(device), torch.nn.BCELoss().to(device)]
d_loss = torch.nn.BCELoss().to(device)
g_param = list(g_model.parameters())
d_param = list(d_model.parameters())
g_optim = torch.optim.Adam(lr=1e-3, params=g_param)
d_optim = torch.optim.Adam(lr=1e-3, params=d_param)
os.mkdir(g_ckpt_path)
os.mkdir(d_ckpt_path)
g_ckpt_cb = ModelCheckPoint(g_ckpt_path, 'deeper_unet')
d_ckpt_cb = ModelCheckPoint(d_ckpt_path, 'discriminator')
trainer = Deep_GAN_Trainer(g_model, d_model, g_loss, d_loss, g_optim, d_optim,
                           g_callbacks=[g_ckpt_cb], d_callbacks=[d_ckpt_cb])
print('load Trainer')
trainer.train(epochs, train_dataloader, test_dataloader)
