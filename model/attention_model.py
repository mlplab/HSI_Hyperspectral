# coding: UTF-8


import torch
from torchsummary import summary
from .layers import Attention_HSI_prior_block


class Attention_HSI_Model(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, block_num=9, **kwargs):
        super(Attention_HSI_Model, self).__init__()

        mode = kwargs['mode']
        ratio = kwargs.get('ratio')
        if ratio is None:
            ratio = 2
        self.activation = kwargs.get('activation')
        self.output_norm = kwargs.get('output_norm')
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        self.start_shortcut = torch.nn.Identity()
        hsi_block = []
        residual_block = []
        # shortcut_block = []
        for _ in range(block_num):
            hsi_block.append(Attention_HSI_prior_block(output_ch, output_ch, activation='relu', ratio=ratio, mode=mode))
            residual_block.append(torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0))
            # shortcut_block.append(torch.nn.Identity())
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Sequential(*residual_block)
        # self.shortcut_block = torch.nn.Sequential(*shortcut_block)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        # self.ita = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        x = self.start_conv(x)
        x_in = self.start_shortcut(x)
        for hsi, residual in zip(self.hsi_block, self.residual_block):
            x_hsi = hsi(x)
            x_residual = residual(x)
            # x = x_in + self.ita * x_hsi + x_residual
            x = x_in + x_hsi + x_residual
        return self._output_norm_fn(self.output_conv(x))

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


if __name__ == '__main__':

    mode = 'GVP'
    model = Attention_HSI_Model(1, 31, 64, 9, activation='relu', mode=mode)
    summary(model, (1, 64, 64))
