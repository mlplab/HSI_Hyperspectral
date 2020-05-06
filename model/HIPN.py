# coding: utf-8
'''
Hyperspectral Image Prior Network Model
'''


import torch
from torchsummary import summary
from layers import swish, mish, HSI_prior_block


class HIPN(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, ratio=8, block_num=9, activation='relu', output_norm=None):
        super(HIPN, self).__init__()
        self.activation = activation
        self.output_norm = output_norm
        # k = 64
        k = feature
        self.start_conv = torch.nn.Conv2d(input_ch, k, 3, 1, 1)
        hsi_prior_block = []
        residual_block = []
        for block in range(block_num):
            hsi_prior_block.append(HSI_prior_block(k, k, ratio=ratio))
            residual_block.append(torch.nn.Conv2d(k, k, 3, 1, 1))
        self.hsi_prior_block = torch.nn.Sequential(*hsi_prior_block)
        self.residual_block = torch.nn.Sequential(*residual_block)
        self.output_conv = torch.nn.Conv2d(k, output_ch, 1, 1, 0)

    def forward(self, x):

        x = self.start_conv(x)
        x_start = x
        for hsi_prior_block, residual_block in zip(self.hsi_prior_block, self.residual_block):
            x_hsi = hsi_prior_block(x)
            x_res = residual_block(x)
            x = x_res + x_start + x_hsi
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


class HSI_Network(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, block_num=9, activation='relu', output_norm=None):
        super(HSI_Network, self).__init__()
        self.activation = activation
        self.output_norm = output_norm
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        self.start_shortcut = torch.nn.Identity()
        hsi_block = []
        residual_block = []
        shortcut = []
        for _ in range(block_num):
            hsi_block.append(HSI_prior_block(output_ch, output_ch, feature=feature))
            residual_block.append(torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0))
            shortcut.append(torch.nn.Identity())
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Sequential(*residual_block)
        self.shortcut = torch.nn.Sequential(*shortcut)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        h = self.start_shortcut(x)
        for hsi_block, residual_block, shortcut in zip(self.hsi_block, self.residual_block, self.shortcut):
            x_hsi = hsi_block(x)
            x_res = residual_block(x)
            x_shortcut = shortcut(h)
            x = x_res + x_shortcut + x_hsi
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


if __name__ == '__main__':

    input_ch = 1
    output_ch = 31
    model = HSI_Network(input_ch, output_ch, block_num=9)
    summary(model, (input_ch, 64, 64))
