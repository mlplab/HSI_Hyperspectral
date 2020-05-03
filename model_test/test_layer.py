# coding: utf-8


import torch


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class Swish(torch.nn.Module):

    def forward(self, x):
        return swish(x)


class Mish(torch.nn.Module):

    def forward(self, x):
        return mish(x)


class Base_Model(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Base_Model, self).__init__()
        keys = kwargs.keys()
        self.activation = None
        if 'activation' in keys:
            self.activation = kwargs['activation']
        self.output_norm = None
        if 'output_norm' in keys:
            self.output_norm = kwargs['output_norm']

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        else:
            return x

    def _output_norm(self, x):
        if self.outpt_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.outpt_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


class Conv2d_Block(Base_Model):

    def __init__(self, input_ch, output_ch, kernel_size, *args, stride=1, padding=0, groups=1, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        keys = kwargs.keys()
        self.norm = None
        self.conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=padding, groups=groups)
        if 'norm' is True:

    def forward(self, x):
        return self._activation_fn(self.conv(x))


class Res_Block


if __name__ == '__main__':

    x = torch.rand((1, 32, 64, 64))
    print(x.max(), x.min())
    conv = Conv2d(32, 64, 3, activation='relu')
    y = conv(x)
    print(y.max(), y.min())
