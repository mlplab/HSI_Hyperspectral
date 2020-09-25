# coding: utf-8


import torch
from torchsummary import summary
from .layers import Ghost_layer, Ghost_Bottleneck, swish, mish


class Ghost_HSCNN(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_num=64, layer_num=9, **kwargs):
        super(Ghost_HSCNN, self).__init__()
        self.activation = kwargs.get('activation', 'relu')
        self.se_flag = kwargs.get('se_flag', False)
        self.start_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.ghost_layers = torch.nn.ModuleList([Ghost_layer(feature_num, feature_num) for _ in range(layer_num)])
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            return x

    def forward(self, x):

        x = self.start_conv(x)
        x_in = x
        for ghost_layer in self.ghost_layers:
            x = self._activation_fn(ghost_layer(x))
        output = self.output_conv(x + x_in)
        return output


class Ghost_HSCNN_Bneck(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_num=64, layer_num=9, **kwargs):
        super(Ghost_HSCNN_Bneck, self).__init__()
        self.activation = kwargs.get('activation', 'relu')
        ratio = kwargs.get('ratio', 2)
        se_flag = kwargs.get('se_flag', False)
        self.start_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.ghost_layers = torch.nn.ModuleList([Ghost_Bottleneck(feature_num, feature_num // ratio, feature_num, se_flag=se_flag) for _ in range(layer_num)])
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            return x

    def forward(self, x):

        x = self.start_conv(x)
        x_in = x
        for ghost_layer in self.ghost_layers:
            x = self._activation_fn(ghost_layer(x))
        output = self.output_conv(x + x_in)
        return output



if __name__ == '__main__':

    model = Ghost_HSCNN_Bneck(1, 31, ratio=2)
    summary(model, (1, 64, 64))
    model = Ghost_HSCNN(1, 31)
    summary(model, (1, 64, 64))