# coding: utf-8


import torch
from torchsummary import summary
from model.layers import Attention_HSI_prior_block
from model.layers import HSI_prior_block
from model.layers import My_HSI_network


if __name__ == '__main__':

    model = Attention_HSI_prior_block(31, 31)
    summary(model, (31, 64, 64))
    del model
    model = HSI_prior_block(31, 31)
    summary(model, (31, 64, 64))
    del model
    model = My_HSI_network(31, 31)
    summary(model, (31, 64, 64))
    optim = torch.optim.Adam(model, params=list(model.parameters()))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=.1)
