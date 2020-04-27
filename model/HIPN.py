# coding: utf-8
'''
Hyperspectral Image Prior Network Model
'''


import torch
from torchsummary import summary
from .layers import swish, mish, HSI_prior_network


class HIPN(torch.nn.Module):

    def __init__(self, input_ch, output_ch, block_num=9, activation='relu', output_norm='sigmoid'):
        super(HIPN, self).__init__()
        # start_ch = 64
        self.activation = activation
        self.output_norm = output_norm
        k = 64
        self.start_conv = torch.nn.Conv2d(input_ch, k, 3, 1, 1)
        hsi_prior_block = []
        residual_block = []
        for block in range(block_num):
            hsi_prior_block.append(HSI_prior_network(k, k))
            residual_block.append(torch.nn.Conv2d(k, k, 3, 1, 1))
        self.hsi_prior_block = torch.nn.Sequential(*hsi_prior_block)
        self.residual_block = torch.nn.Sequential(*residual_block)
        self.output_conv = torch.nn.Conv2d(k, output_ch, 3, 1, 1)

    def forward(self, x):

        x = self.start_conv(x)
        x_start = x
        for hsi_prior_block, residual_block in zip(self.hsi_prior_block, self.residual_block):
            x_hsi = hsi_prior_block(x)
            x_res = residual_block(x)
            x = x_res + x + x_hsi
            # x = torch.cat([x_start, s_hsi, x_res], dim=1)
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


if __name__ == '__main__':

    model = HIPN(32, 31)
    summary(model, (32, 64, 64))