# coding: utf-8

import os
import torch
# import torchvision
from trainer import Trainer
# from model.HIPN import HSI_Network
from model.test_model import Attention_HSI_Model
from data_loader import PatchMaskDataset
# from utils import RandomCrop, RandomHorizontalFlip
from utils import ModelCheckPoint


batch_size = 1
epochs = 5
# data_len = batch_size * 10

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


data_name = 'CAVE'
img_path = 'dataset/'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
ckpt_path = 'ckpt'
# drive_path = '/content/drive/My Drive/auto_colorization/'

train_transform = (RandomHorizontalFlip(), torchvision.transform.ToTensor())
# train_transform = None
# test_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, os.path.join(img_path, 'mask_data'), tanh=False, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, os.path.join(img_path, 'mask_data'), tanh=False, transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
model = Attention_HSI_Model(1, 31).to(device)
criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
ckpt_cb = ModelCheckPoint(os.path.join(ckpt_path, data_name), 'HSI_Network', mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, callbacks=[ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
