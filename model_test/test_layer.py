# coding: utf-8


import torch


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class Swish(torch.nn.Module):

    def forward(self, x):
        return swish(x)


class Mish(torch.nn.Module):

    def forward(self, x):
        return mish(x)


'''
class Base_Model(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, **kwargs):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
'''


class DW_PT_Conv(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, activation='relu'):
        super(DW_PT_Conv, self).__init__()
        self.activation = activation
        self.depth = torch.nn.Conv2d(input_ch, input_ch, kernel_size, 1, 1, groups=input_ch)
        self.point = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        else:
            return x

    def forward(self, x):
        x = self.depth(x)
        x = self._activation_fn(x)
        x = self.point(x)
        x = self._activation_fn(x)
        return x


class HSI_prior_network(torch.nn.Module):

    def __init__(self, input_ch, output_ch, activation='relu', ratio=4):
        super(HSI_prior_network, self).__init__()
        self.activation = activation
        self.spatial_1 = torch.nn.Conv2d(input_ch, int(input_ch * ratio), 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(int(input_ch * ratio), output_ch, 3, 1, 1)
        self.shortcut = torch.nn.Identity()
        self.spectral = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def forward(self, x):
        x_in = self.shortcut(x)
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        # h = self._activation_fn(h)
        # x = torch.cat([x, x_in], dim=1)
        x = h + x_in
        x = self.spectral(x)
        # x = self._activation_fn(x)
        return x


class My_HSI_network(torch.nn.Module):

    def __init__(self, input_ch, output_ch, activation='relu', ratio=4):
        super(My_HSI_network, self).__init__()
        self.activation = activation
        self.spatial_1 = DW_PT_Conv(input_ch, int(input_ch * ratio), 3, activation=None)
        self.spatial_2 = DW_PT_Conv(int(input_ch * ratio), input_ch, 3, activation=None)
        self.shortcut = torch.nn.Identity()
        self.spectral = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        else:
            return x

    def forward(self, x):
        x_in = self.shortcut(x)
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        # h = self._activation_fn(h)
        # x = torch.cat([x, x_in], dim=1)
        x = h + x_in
        x = self.spectral(x)
        # x = self._activation_fn(x)
        return x


class Global_Variance_Pooling(torch.nn.Module):

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        avg_x = torch.nn.functional.avg_pool2d(x, kernel=(h, w))
        var_x = torch.nn.functional.avg_pool2d((x - avg_x) ** 2, kernel=(h, w))
        return var_x


class RAM(torch.nn.Module):

    def __init__(self, input_ch, output_ch, raito=None):
        super(RAM, self).__init__()

        if raito is None:
            self.raito = 2
        else:
            self.raito = raito
        self.spatial_attn = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1, groups=input_ch)
        self.spectral_pooling = Global_Variance_Pooling()
        self.spectral_Linear = torch.nn.Linear(input_ch, input_ch // raito)
        self.spectral_attn = torch.nn.Linear(input_ch // raito, output_ch)

    def forward(self, x):
        spatial_attn = self.spectral_attn(x)
        spectral_pooling = self.spectral_pooling(x)
        spectral_linear = torch.relu(self.spectral_Linear(spectral_pooling))
        spectral_attn = self.spectral_attn(spectral_linear)

        attn_output = torch.sigmoid(spatial_attn + spectral_attn)
        output = attn_output * x
        return output
