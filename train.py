# coding: utf-8

import os
import torch
import torchvision
from trainer import Trainer
# from model.attention_model import Attention_HSI_Model
from model.HSCNN import HSCNN
from model.autoencoder import AutoEncoder
from data_loader import PatchMaskDataset, PatchEvalDataset, PatchMaskDataset_AutoEncoder
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint


batch_size = 1
epochs = 5


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


data_name = 'My_CAVE'
img_path = f'../SCI_dataset/{data_name}'
model_name = 'AutoEncoder'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
ckpt_path = '../SCI_ckpt'


train_transform = (RandomHorizontalFlip(), RandomRotation(), torchvision.transforms.ToTensor())
test_transform = None
# train_dataset = PatchMaskDataset(train_path, mask_path, tanh=False, transform=train_transform)
pre_train_dataset = PatchMaskDataset_AutoEncoder(train_path, mask_path=None, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(pre_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
pre_test_dataset = PatchMaskDataset_AutoEncoder(test_path, mask_path=None, tanh=False, transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(pre_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# model = HSCNN(1, 31).to(device)
model = AutoEncoder(31, 31).to(device)
criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 50, .1)


ckpt_cb = ModelCheckPoint(os.path.join(ckpt_path, data_name), model_name, mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, scheduler=scheduler, callbacks=[ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
