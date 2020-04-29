# coding: utf-8


import torch
from torchsummary import summary
import layers


device = 'cpu'


class Test_Model(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature=64, block=9, ratio=8, **kwargs):
        super(Test_Model, self).__init__()

        keys = kwargs.keys()
        if 'activation' not in keys:
            self.activation = None
        else:
            self.activation = kwargs['activation']
        if 'output_norm' not in keys:
            self.output_norm = None
        else:
            self.output_norm = kwargs['output_norm']
        self.start_conv = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        hsi_block = [layers.My_HSI_network(feature, feature, ratio=ratio) for _ in range(block)]
        residual_block = [torch.nn.Conv2d(feature, feature, 3, 1, 1) for _ in range(block)]
        shortcut_block = [torch.nn.Identity() for _ in range(block)]
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Sequential(*residual_block)
        self.shortcut_block = torch.nn.Sequential(*shortcut_block)
        self.output_conv = torch.nn.Conv2d(feature, output_ch, 1, 1, 0)

    def forward(self, x):

        x = self.start_conv(x)
        x_start = x
        for hsi_block, residual_block, shortcut_block in zip(self.hsi_block, self.residual_block, self.shortcut_block):
            x_hsi = hsi_block(x)
            x_res = residual_block(x)
            x_start = shortcut_block(x_start)
            x = x_res + x_start + x_hsi
            # x = torch.cat([x_start, s_hsi, x_res], dim=1)
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return layers.swish(x)
        elif self.activation == 'mish':
            return layers.mish(x)
        else:
            return x

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


if __name__ == '__main__':

    model = Test_Model(1, 31, feature=64, block=9, ratio=8).to(device)
    summary(model, (1, 64, 64))
