# coding: utf-8


import torch
from torchsummary import summary


class HSCNN(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature=64, layer_num=9, **kwargs):
        super(HSCNN, self).__init__()
        self.activation = None
        self.output_norm = None
        if 'activation' in kwargs:
            self.activation = kwargs['activation']
        if 'output_norm' in kwargs:
            self.output_norm = kwargs['output_norm']
        # self.residual_shortcut = torch.nn.Identity()
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.patch_extraction = torch.nn.Conv2d(output_ch, feature, 3, 1, 1)
        feature_map = [torch.nn.Conv2d(feature, feature, 3, 1, 1) for _ in range(layer_num - 1)]
        self.feature_map = torch.nn.Sequential(*feature_map)
        self.residual_conv = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)

    def forward(self, x):

        x = self.start_conv(x)
        # x_in = self.residual_shortcut(x)
        x_in = x
        x = self._activation_fn(self.patch_extraction(x))
        for feature_map in self.feature_map:
            x = self._activation_fn(feature_map(x))
        output = self.residual_conv(x) + x_in
        return output

    def show_features(self, x, layer_num=0, output_layer=True, activation=True):

        # initialize result
        result = []
        if isinstance(layer_num, int):
            layer_num = [layer_num]
        j = 0
        layer_num = set(layer_num)
        layer_nums = []
        layer_nums = [True if i in layer_num else False for i in range(1, len(self.feature_map) + 1)]

        # add start_conv
        x = self.start_conv(x)
        if layer_nums[0]:
            result.append(x)
        x_in = x

        # add feature map
        for i, feature_map in enumerate(self.feature_map):
            x = feature_map(x)
            # if activation is True:
            x = self._activation_fn(x)
            if layer_nums[i]:
                result.append(x)

        output = self.residual_conv(x + x_in)
        if output_layer:
            result.append(output)
        return result

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        if self.activation == 'leaky':
            return torch.nn.functional.leaky_relu(x)
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

    model = HSCNN(1, 31)
    summary(model, (1, 64, 64))