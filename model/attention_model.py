# coding: UTF-8


import torch
from torchsummary import summary
from layers import swish, mish, leaky_relu, Attention_HSI_prior_block


class Attention_HSI_Model_2(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, block_num=9, **kwargs):
        super(Attention_HSI_Model_2, self).__init__()
        self.activation = kwargs.get('activation')
        self.output_norm = kwargs.get('output_norm')
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        hsi_block = []
        residual_block = []
        for _ in range(block_num):
            hsi_block.append(Attention_HSI_prior_block(output_ch, output_ch, activation='relu'))
            residual_block.append(torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0))
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Sequential(*residual_block)

    def forward(self, x):
        x = self.start_conv(x)
        x_in = x
        for hsi, residual in zip(self.hsi_block, self.residual_block):
            x_hsi = hsi(x)
            x_residual = residual(x)
            x += x_in + x_hsi + x_residual
        return self._output_norm_fn(x)

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return leaky_relu(x)
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

    model = Attention_HSI_Model_2(1, 31, 64, 3, activation='relu')
    summary(model, (1, 64, 64))
