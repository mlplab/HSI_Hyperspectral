# coding: UTF-8


import torch
from model.layers import Base_Module


class Conv_Model(Base_Module):

    def __init__(self, input_ch, output_ch, **kwargs):
        super(Conv_Model, self).__init__()
        self.activation = kwargs.get('activation')
        self.conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        return self._activation_fn(self.conv(x))


x = torch.rand((1, 32, 32, 32))
conv = Conv_Model(32, 32, activation='swish')
y = conv(x)
print(y.max(), y.min())
