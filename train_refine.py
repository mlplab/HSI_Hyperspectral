# coding: utf-8


import os
import torch
# import torchvision
from trainer import Trainer
from model.unet_copy import UNet_Res
from data_loader import RefineEvaluateDataset
from utils import ModelCheckPoint


crop = 256
batch_size = 1
epochs = 5
# data_len = batch_size * 10

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

img_path = 'dataset/'
output_train_path = os.path.join(img_path, 'output_ch_data')
output_test_path = os.path.join(img_path, 'output_test')
train_path = os.path.join(img_path, 'train_patch')
test_path = os.path.join(img_path, 'test_patch')
model_path = 'refine_unet'
ckpt_path = os.path.join('ckpt', model_path)
# drive_path = '/content/drive/My Drive/auto_colorization/'

train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
# train_dataset = RefineDataset(output_train_path, train_path, transform=None)
train_dataset = RefineEvaluateDataset(output_train_path, train_path, None, None)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# test_dataset = RefineDataset(output_test_path, test_path, transform=None)
test_dataset = RefineEvaluateDataset(output_test_path, test_path, None, None)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
model = UNet_Res(1, 1).to(device)
criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
ckpt_cb = ModelCheckPoint(ckpt_path, 'refile_unet', mkdir=False)
trainer = Trainer(model, criterion, optim, device, [ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
