# coding: utf-8


import torch


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class Base_Model(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, **kwargs):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)