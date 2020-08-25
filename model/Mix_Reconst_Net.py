# coding: UTF-8


import torch
from torchsummary import summary
from .layers import Mix_Conv, Mix_SS_Layer


class Mix_Reconst_Net(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chunks, *args, block_num=9, feature_num=64, **kwargs):
        super(Mix_Reconst_Net, self).__init__()

        self.activation = kwargs.get('activation')
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.mix_ss_layers = torch.nn.ModuleList([Mix_SS_Layer(output_ch,
                                                               output_ch, chunks,
                                                               feature_num=feature_num) for _ in range(block_num)])
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        for mix_ss_layer in self.mix_ss_layers:
            x = mix_ss_layer(x)
        return self.output_conv(x)


if __name__ == '__main__':

    model = Mix_Reconst_Net(1, 31, 4, group_num=1)
    summary(model, (1, 48, 48))
