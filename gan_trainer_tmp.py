# coding: utf-8


'''
import numpy as np
from tqdm import tqdm
import torch
# from torchsummary import summary
# from model.encoder_decoder import CAE
from trainer import Trainer
# from model.unet import UNet
# from model.discriminator import Discriminator
# from data_loader import HyperSpectralDdataset
from utils import psnr
# from utils import RandomCrop, RandomHorizontalFlip, ModelCheckPoint
'''
import os
from tqdm import tqdm
from datetime import datetime
import torch
import torchvision
from trainer import Trainer
# from gan_trainer import GAN_Trainer
from model.unet import UNet
from model.discriminator import Discriminator
from data_loader import HyperSpectralDdataset
from utils import RandomCrop, RandomHorizontalFlip, ModelCheckPoint, psnr


class GAN_Trainer(Trainer):

    def __init__(self, g_model, d_model, g_criterion, d_criterion, g_optim, d_optim,
                 batch_size, device='cpu', d_callbacks=None, g_callbacks=None):

        self.g_model = g_model
        self.d_model = d_model
        self.g_criterion = g_criterion
        self.d_criterion = d_criterion
        self.g_optim = g_optim
        self.d_optim = d_optim
        # self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.g_optim, mode='min', factor=.2, patience=2, verbose=True, min_lr=1e-8)
        # self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.d_optim, mode='min', factor=.2, patience=2, verbose=True, min_lr=1e-8)
        self.batch_size = batch_size
        self.device = device
        self.d_callbacks = d_callbacks
        self.g_callbacks = g_callbacks

    def train(self, epochs, train_dataloader, val_dataloader, init_epoch=None):

        if init_epoch is None:
            init_epoch = 0
        elif isinstance(init_epoch, int):
            assert 'Please enter int to init_epochs'

        # _, columns = os.popen('stty size', 'r').read().split()
        # columns = int(columns) // 2
        columns = 200

        for epoch in range(init_epoch, epochs):
            ###################################################################
            # Train
            ###################################################################
            dt_now = datetime.now()
            print(dt_now)
            self.g_model.train()
            self.d_model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            with tqdm(train_dataloader, desc=f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}', ncols=columns, unit='step') as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    g_loss, d_loss, g_mse_loss = self._step(inputs, labels)
                    train_loss.append(g_mse_loss.item())
                    psnr_show = psnr(g_mse_loss)
                    self._step_show(pbar, mode, epoch, g_loss, psnr_show)
                    torch.cuda.empty_cache()
            ###################################################################
            # Val
            ###################################################################
            '''
            self.g_model.eval()
            self.d_model.eval()
            with tqdm(val_dataloader, desc=f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}', ncols=columns, unit='step') as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    with torch.no_grad():
                        g_loss, d_loss, g_mse_loss = self._step(inputs, labels)
                    train_loss.append(g_mse_loss.item())
                    psnr_show = psnr(g_mse_loss)
                    self._step_show(pbar, mode, epoch, g_loss, psnr_show)
                    torch.cuda.empty_cache()
            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)
            if self.g_callbacks:
                for callback in self.g_callbacks:
                    callback.callback(self.g_model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True, device=self.device)
            if self.d_callbacks:
                for callback in self.d_callbacks:
                    callback.callback(self.d_model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True, device=self.device)
            '''
            # self.scheduler.step(val_loss)
            print('-' * int(columns))
        return self

    def _step(self, inputs, labels, train=True):
        batch_size = inputs.size()[0]
        output_real = torch.ones(batch_size, 1).to(self.device)
        output_fake = torch.zeros(batch_size, 1).to(self.device)
        ####################################
        g_output = self.g_model(inputs)
        ####################################
        d_labels = torch.cat([inputs[:, 0:1, :, :], labels], dim=1)
        d_output = torch.cat([inputs[:, 0:1, :, :], g_output], dim=1)
        d_real = self.d_model(d_labels)
        d_real_loss = self.d_criterion(d_real, output_real)
        d_fake = self.d_model(d_output)
        d_fake_loss = self.d_criterion(d_fake, output_fake)
        d_loss = d_real_loss + d_fake_loss
        if train is True:
            self.g_optim.zero_grad()
            self.d_optim.zero_grad()
            d_loss.backward(retain_graph=True)
            self.d_optim.step()
        ####################################
        g_mse_loss = self.g_criterion[0](g_output, labels)
        g_fake = self.d_model(d_output)
        g_bce_loss = self.g_criterion[1](g_fake, output_real)
        g_loss = g_mse_loss + g_bce_loss
        if train is True:
            self.d_optim.zero_grad()
            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()
        ####################################
        return g_loss, d_loss, g_mse_loss


if __name__ == '__main__':

    crop = 96
    batch_size = 1
    epochs = 5
    # data_len = batch_size * 10

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.backends.cudnn.benchmark = True

    img_path = 'dataset/'
    train_path = os.path.join(img_path, 'train')
    # test_path = os.path.join(img_path, 'test')
    # drive_path = '/content/drive/My Drive/auto_colorization/'

    train_list = os.listdir(train_path)
    # test_list = os.listdir(test_path)
    transform = (RandomCrop(crop), RandomHorizontalFlip(),
                 torchvision.transforms.ToTensor())
    # test_transform = (RandomHorizontalFlip(), torchvision.transform.ToTensor())
    train_dataset = HyperSpectralDdataset(
        train_path, os.path.join(img_path, 'mask.mat'), transform=transform)
    # test_dataset = HyperSpectralDdataset(
    #     train_path, os.path.join(img_path, 'mask.mat'), transform=test_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    g_model = UNet(25, 24).to(device)
    d_model = Discriminator([32, 64, 128], (crop, crop), 25).to(device)
    g_loss = [torch.nn.MSELoss().to(device), torch.nn.BCELoss().to(device)]
    d_loss = torch.nn.BCELoss().to(device)
    g_param = list(g_model.parameters())
    d_param = list(d_model.parameters())
    g_optim = torch.optim.Adam(lr=1e-3, params=g_param)
    d_optim = torch.optim.Adam(lr=1e-3, params=d_param)
    g_ckpt_cb = ModelCheckPoint('g_ckpt', 'unet')
    d_ckpt_cb = ModelCheckPoint('d_ckpt', 'discriminator')
    trainer = GAN_Trainer(g_model, d_model, g_loss, d_loss, g_optim, d_optim, batch_size,
                          device=device, g_callbacks=[g_ckpt_cb], d_callbacks=[d_ckpt_cb])
    trainer.train(epochs, train_dataloader, None)
