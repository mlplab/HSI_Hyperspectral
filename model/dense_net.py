# coding: utf-8


import torch
from torchsummary import summary
from .layers import HSI_prior_block


class Dense_HSI_prior_Network(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, block_num=9, **kwargs):
        super(Dense_HSI_prior_Network, self).__init__()
        self.output_norm = kwargs.get('output_norm')
        activation = kwargs.get('activation')
        feature = kwargs.get('feature')
        if feature is None:
            feature = 64
        stack_num = output_ch * block_num
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.start_shortcut = torch.nn.Identity()
        # self.residual_block = torch.nn.ModuleList([torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1) for _ in range(block_num)])
        self.residual_block = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        self.hsi_block = torch.nn.ModuleList([HSI_prior_block(output_ch, output_ch, feature=feature, activation=activation) for _ in range(block_num)])
        self.dense_conv = torch.nn.ModuleList([torch.nn.Conv2d(output_ch + output_ch, output_ch, 3, 1, 1) for _ in range(block_num)])
        # self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        self.output_conv = torch.nn.Conv2d(stack_num, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        x_in = self.start_shortcut(x)
        x_stack = []
        for hsi, dense_conv in zip(self.hsi_block, self.dense_conv):
            x_res = self.residual_block(x)
            x_hsi = hsi(x)
            x = x_res + x_hsi
            x = torch.cat((x, x_in), dim=1)
            x = dense_conv(x)
            x_stack.append(x)
        x_stack = torch.cat(x_stack, dim=1)
        return self._output_norm(self.output_conv(x_stack))

    def _output_norm(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


if __name__ == '__main__':

    input_ch = 1
    output_ch = 31
    model = Dense_HSI_prior_Network(input_ch, output_ch, block_num=9)
    summary(model, (input_ch, 64, 64))
    x = torch.rand((1, input_ch, 64, 64))
    y = model(x)
    print(y.shape)
