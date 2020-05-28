# coding: utf-8

import os
import argparse
import torch
import torchvision
from trainer import Trainer
from model.attention_model import Attention_HSI_Model
from model.HSCNN import HSCNN
from model.HIPN import HSI_Network
from data_loader import PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=150, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--model_name', '-m', default='HSCNN', type=str, help='Model Name')
args = parser.parse_args()


batch_size = args.batch_size
epochs = args.epochs


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


data_name = args.dataset
img_path = f'../SCI_dataset/My_{data_name}'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
ckpt_path = os.path.join('../SCI_ckpt', f'{data_name}_0529')
train_transform = (RandomHorizontalFlip(), RandomRotation(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, mask_path, tanh=False, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, mask_path, tanh=False, transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model_name = args.model_name
# model_name = 'Attention_HSI_Model'
if model_name == 'HSCNN':
    model = HSCNN(1, 31)
elif model_name == 'HSI_Network':
    model = HSI_Network(1, 31)
elif model_name == 'Attention_HSI':
    model = Attention_HSI_Model(1, 31, mode=None, ratio=4)
elif model_name == 'Attention_HSI_GAP':
    model = Attention_HSI_Model(1, 31, mode='GAP', ratio=4)
elif model_name == 'Attention_HSI_GVP':
    model = Attention_HSI_Model(1, 31, mode='GVP', ratio=4)
else:
    assert 'Enter Model Name'


model.to(device)
criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 50, .1)


summary(model, (1, 64, 64))
print(model_name)


ckpt_cb = ModelCheckPoint(os.path.join(ckpt_path, data_name), model_name, mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, scheduler=scheduler, callbacks=[ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
