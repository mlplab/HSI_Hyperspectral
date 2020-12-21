# coding: utf-8


import torch
from torchsummary import summary
from .layers import HSI_prior_block, swish


class Dense_HSI_prior_Network(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, block_num=9, **kwargs):
        super(Dense_HSI_prior_Network, self).__init__()
        self.output_norm = kwargs.get('output_norm')
        activation = kwargs.get('activation')
        feature = kwargs.get('feature')
        self.split_num = kwargs.get('split_num')
        if self.split_num is None:
            self.split_num = 1
        self.block_num = block_num
        if feature is None:
            feature = 64
        stack_num = output_ch * block_num
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.residual_block = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        self.hsi_block = torch.nn.ModuleList([HSI_prior_block(output_ch, output_ch, feature=feature, activation=activation) for _ in range(block_num)])
        if self.split_num == 1:
            self.split_conv = torch.nn.ModuleList([torch.nn.Identity() for _ in range(self.block_num)])
            self.output_conv = torch.nn.Conv2d(output_ch * self.block_num, output_ch, 1, 1, 0)
        else:
            split_conv = [torch.nn.Conv2d(output_ch * self.split_num, output_ch, 1, 1, 0) for _ in range(self.block_num // self.split_num)]
            if block_num % self.split_num > 0:
                split_conv.append(torch.nn.Conv2d(output_ch * (block_num % self.split_num), output_ch, 1, 1, 0))
            self.split_conv = torch.nn.ModuleList(split_conv)
            # self.output_conv = torch.nn.Conv2d(output_ch * (self.block_num // self.split_num + self.block_num % self.split_num), output_ch, 1, 1, 0)
            self.output_conv = torch.nn.Conv2d(output_ch * len(self.split_conv), output_ch, 1, 1, 0)
        # self.output_conv = torch.nn.Conv2d(output_ch * self.block_num % self.split_num, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        x_in = x
        x_stack = []
        x_pre_stack = []
        j = 0
        for i, hsi in enumerate(self.hsi_block):
            x_res = self.residual_block(x)
            x_hsi = hsi(x)
            x = x_hsi + x_res + x_in
            x_pre_stack.append(x)
            if (i + 1) % self.split_num == 0:
                x_pre_stack = torch.cat(x_pre_stack, dim=1)
                x_pre = swish(self.split_conv[j](x_pre_stack))
                x_stack.append(x_pre)
                x_pre_stack = []
                j += 1
        if len(x_pre_stack) > 0:
            if len(x_pre_stack) == 1:
                x_pre_stack = x_pre_stack[0]
            else:
                x_pre_stack = torch.cat(x_pre_stack, dim=1)
            x_pre = swish(self.split_conv[-1](x_pre_stack))
            x_stack.append(x_pre)
        x_stack = torch.cat(x_stack, dim=1)
        x_output = self.output_conv(x_stack)
        return x_output

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
    block_num = 5
    split_num = 1
    model = Dense_HSI_prior_Network(input_ch, output_ch,
                                    block_num=block_num, split_num=split_num)
    summary(model, (input_ch, 64, 64))
    x = torch.rand((1, input_ch, 64, 64))
    y = model(x)
    print(y.shape)
