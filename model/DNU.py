# coding: UTF-8


import torch
from torchsummary import summary
from .layers import DNU_Block


class DNU(torch.nn.Module):

    def __init__(self, input_ch, output_ch, Cu, *args, feature_num=64, block_num=11, deta=.04, eta=.8, omega=.8, **kwargs):
        super(DNU, self).__init__()
        self.DNU_layers = torch.nn.ModuleList([DNU_Block(input_ch=output_ch, 
                                                         output_ch=output_ch, 
                                                         feature_num=feature_num, 
                                                         deta=deta, eta=eta, omega=omega) for _ in range(block_num))])

    def forward(self, x, Cu):

        x0 = x
        for i, layer in enumerate(self.DNU_layers):
            x = layer(x, x0, Cu)

        return x
