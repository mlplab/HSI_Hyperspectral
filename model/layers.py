# coding: utf-8


import numpy as np
import torch


def split_layer(output_ch, chunks):
    split = [np.int(np.ceil(output_ch / chunks)) for _ in range(chunks)]
    split[chunks - 1] += output_ch - sum(split)
    return split


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class FReLU(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride=1, **kwargs):
        super(FReLU, self).__init__()
        self.depth = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        depth = self.depth(x)
        return torch.max(x, depth)


class Swish(torch.nn.Module):

    def forward(self, x):
        return swish(x)


class Mish(torch.nn.Module):

    def forward(self, x):
        return mish(x)


class Base_Module(torch.nn.Module):

    def __init__(self):
        super(Base_Module, self).__init__()
        # self.activation =
        # if self.activation == 'swish':
        #     self._activation_fn = Swish()
        # elif self.activation == 'mish':
        #     self._activation_fn = Mish()
        # elif self.activation == 'leaky' or self.activation == 'leaky_relu':
        #     self._activation_fn = torch.nn.LeakyReLU()
        # elif self.activation == 'frelu':
        #     self._activation_fn = FReLU(frelu_input_ch, frelu_output_ch)
        # elif self.activation == 'relu':
        #     self._activation_fn = torch.nn.ReLU()
        # else:
        #     self.activation = torch.nn.Sequential()

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        # elif self.activation == 'frelu':
        #     return FReLU(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            return x
class SAMLoss(torch.nn.Module):

    def forward(self, x, y):
        x_sqrt = torch.norm(x, dim=1)
        y_sqrt = torch.norm(y, dim=1)
        xy = torch.sum(x * y, dim=1)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torch.acos(metrics)
        return torch.mean(angle)


class MSE_SAMLoss(torch.nn.Module):

    def __init__(self, alpha=.5, beta=.5, mse_ratio=1., sam_ratio=.01):
        super(MSE_SAMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = torch.nn.MSELoss()
        self.sam_loss = SAMLoss()
        self.mse_ratio = mse_ratio
        self.sam_ratio = sam_ratio

    def forward(self, x, y):
        return self.alpha * self.mse_ratio * self.mse_loss(x, y) + self.beta * self.sam_ratio * self.sam_loss(x, y)


class Conv_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride, padding, norm=True):
        super(Conv_Block, self).__init__()
        layer = []
        layer.append(torch.nn.Conv2d(input_ch, output_ch,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding))
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(output_ch))
        # layer.append(torch.nn.ReLU())
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return torch.nn.functional.relu(self.layer(x))


class D_Conv_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, norm=True):
        super(D_Conv_Block, self).__init__()
        layer = [torch.nn.ConvTranspose2d(
            input_ch, output_ch, kernel_size=2, stride=2)]
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(output_ch))
        layer.append(torch.nn.ReLU())
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class SA_Block(torch.nn.Module):

    def __init__(self, input_ch):
        super(SA_Block, self).__init__()
        self.theta = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.phi = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.g = torch.nn.Conv2d(input_ch, input_ch, 1, 1, 0)
        # self.attn = torch.nn.Conv2d(input_ch // 2, input_ch, 1, 1, 0)
        self.sigma_ratio = torch.nn.Parameter(
            torch.zeros(1), requires_grad=True)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        # theta path (first conv block)
        theta = self.theta(x)
        theta = theta.view(batch_size, ch // 8, h * w).permute((0, 2, 1))  # (bs, HW, CH // 8)
        # phi path (second conv block)
        phi = self.phi(x)
        phi = torch.nn.functional.max_pool2d(phi, 2)
        phi = phi.view(batch_size, ch // 8, h * w // 4)  # (bs, CH // 8, HW)
        # attention path (theta and phi)
        attn = torch.bmm(theta, phi)  # (bs, HW, HW // 4)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # g path (third conv block)
        g = self.g(x)
        g = torch.nn.functional.max_pool2d(g, 2)
        # (bs, HW // 4, CH)
        g = g.view(batch_size, ch, h * w // 4).permute((0, 2, 1))
        # attention map (g and attention path)
        attn_g = torch.bmm(attn, g)  # (bs, HW, CH)
        attn_g = attn_g.permute((0, 2, 1)).view(
            batch_size, ch, h, w)  # (bs, CH, H, W)
        return x + self.sigma_ratio * attn_g


class DW_PT_Conv(Base_Module):

    def __init__(self, input_ch, output_ch, kernel_size, activation='relu'):
        super(DW_PT_Conv, self).__init__()
        self.activation = activation
        self.depth = torch.nn.Conv2d(input_ch, input_ch, kernel_size, 1, 1, groups=input_ch)
        self.point = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.depth(x)
        x = self._activation_fn(x)
        x = self.point(x)
        x = self._activation_fn(x)
        return x


class HSI_prior_block(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, activation='relu'):
        super(HSI_prior_block, self).__init__()
        self.activation = activation
        self.spatial_1 = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.spectral = torch.nn.Conv2d(output_ch, input_ch, 1, 1, 0)

    def forward(self, x):
        x_in = x
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


class My_HSI_network(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, activation='relu'):
        super(My_HSI_network, self).__init__()
        self.activation = activation
        self.spatial_1 = DW_PT_Conv(input_ch, feature, 3, activation=None)
        self.spatial_2 = DW_PT_Conv(feature, output_ch, 3, activation=None)
        self.spectral = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x_in = x
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


class RAM(Base_Module):

    def __init__(self, input_ch, output_ch, ratio=None, **kwargs):
        super(RAM, self).__init__()

        if ratio is None:
            self.ratio = 2
        else:
            self.ratio = ratio
        self.activation = kwargs.get('attn_activation')
        self.spatial_attn = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1, groups=input_ch)
        self.spectral_pooling = GVP()
        self.spectral_Linear = torch.nn.Linear(input_ch, input_ch // self.ratio)
        self.spectral_attn = torch.nn.Linear(input_ch // self.ratio, output_ch)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        spatial_attn = self._activation_fn(self.spatial_attn(x))
        spectral_pooling = self.spectral_pooling(x).view(-1, ch)
        spectral_linear = torch.relu(self.spectral_Linear(spectral_pooling))
        spectral_attn = self.spectral_attn(spectral_linear).unsqueeze(-1).unsqueeze(-1)

        # attn_output = torch.sigmoid(spatial_attn + spectral_attn + spectral_pooling.unsqueeze(-1).unsqueeze(-1))
        attn_output = torch.sigmoid(spatial_attn * spectral_attn)
        output = attn_output * x
        return output


class Global_Average_Pooling2d(torch.nn.Module):

    def forward(self, x):
        bs, ch, h, w = x.size()
        return torch.nn.functional.avg_pool2d(x, kernel_size=(h, w)).view(-1, ch)


class GVP(torch.nn.Module):

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        avg_x = torch.nn.functional.avg_pool2d(x, kernel_size=(h, w))
        var_x = torch.nn.functional.avg_pool2d((x - avg_x) ** 2, kernel_size=(h, w))
        return var_x.view(-1, ch)


class SE_block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, **kwargs):
        super(SE_block, self).__init__()
        if 'ratio' in kwargs:
            ratio = kwargs['ratio']
        else:
            ratio = 2
        mode = kwargs.get('mode')
        if mode == 'GVP':
            self.pooling = GVP()
        else:
            self.pooling = Global_Average_Pooling2d()
        self.squeeze = torch.nn.Linear(input_ch, output_ch // ratio)
        self.extention = torch.nn.Linear(output_ch // ratio, output_ch)

    def forward(self, x):
        gap = torch.mean(x, [2, 3], keepdim=True)
        squeeze = self._activation_fn(self.squeeze(gap))
        extention = self.extention(squeeze)
        return torch.sigmoid(extention) * x


class Attention_HSI_prior_block(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, **kwargs):
        super(Attention_HSI_prior_block, self).__init__()
        self.mode = kwargs.get('mode')
        ratio = kwargs.get('ratio')
        if ratio is None:
            ratio = 2
        attn_activation = kwargs.get('attn_activation')
        self.spatial_1 = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.attention = RAM(output_ch, output_ch, ratio=ratio, attn_activation=attn_activation)
        self.spectral = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        if self.mode is not None:
            self.spectral_attention = SE_block(output_ch, output_ch, mode=self.mode, ratio=ratio)
        self.activation = kwargs.get('activation')

    def forward(self, x):
        x_in = x
        h = self._activation_fn(self.spatial_1(x))
        h = self.spatial_2(h)
        h = self.attention(h)
        x = h + x_in
        x = self.spectral(x)
        if self.mode is not None:
            x = self.spectral_attention(x)
        return x


class GroupConv(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chunks, kernel_size, *args, stride=1, **kwargs):
        super(GroupConv, self).__init__()
        self.chunks = chunks
        self.split_input_ch = split_layer(input_ch, chunks)
        self.split_output_ch = split_layer(output_ch, chunks)

        if chunks == 1:
            self.group_conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2)
        else:
            self.group_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.split_input_ch[idx],
                                                                     self.split_output_ch[idx],
                                                                     kernel_size, stride=stride,
                                                                     padding=kernel_size // 2) for idx in range(chunks)])

    def forward(self, x):
        if self.chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.chunk(x, self.chunks, dim=1)
            return torch.cat([group_layer(split_x) for group_layer, split_x in zip(self.group_layers, split)], dim=1)


class Mix_Conv(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chunks, stride=1, **kwargs):
        super(Mix_Conv, self).__init__()

        self.chunks = chunks
        self.split_layer = split_layer(output_ch, chunks)
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.split_layer[idx],
                                                                self.split_layer[idx],
                                                                kernel_size=2 * idx + 3,
                                                                stride=stride,
                                                                padding=(2 * idx + 3) // 2,
                                                                groups=self.split_layer[idx]) for idx in range(chunks)])

    def forward(self, x):
        split = torch.chunk(x, self.chunks, dim=1)
        output = torch.cat([conv_layer(split_x) for conv_layer, split_x in zip(self.conv_layers, split)], dim=1)
        return output


class Group_SE(torch.nn.Module):
    def __init__(self, input_ch, output_ch, chunks, kernel_size, **kwargs):
        super(Group_SE, self).__init__()
        if 'ratio' in kwargs:
            ratio = kwargs['ratio']
        else:
            ratio = 2
        self.activation = kwargs.get('activation')
        feature_num = max(1, output_ch // ratio)
        self.squeeze = GroupConv(input_ch, feature_num, chunks, kernel_size, 1, 0)
        self.extention = GroupConv(feature_num, output_ch, chunks, kernel_size, 1, 0)

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        else:
            return torch.relu(x)

    def forward(self, x):
        gap = torch.mean(x, [2, 3], keepdim=True)
        squeeze = self._activation_fn(self.squeeze(gap))
        extention = self.extention(squeeze)
        return torch.sigmoid(extention) * x


class Mix_SS_Layer(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chunks, *args, stride=1, feature_num=64, group_num=1, **kwargs):
        super(Mix_SS_Layer, self).__init__()
        self.activation = kwargs.get('activation')
        se_flag = kwargs.get('se_flag')
        # self.spatial_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.spatial_conv = GroupConv(input_ch, feature_num, group_num, kernel_size=3, stride=1)
        self.mix_conv = Mix_Conv(feature_num, feature_num, chunks)
        if se_flag:
            self.se_block = Group_SE(feature_num, feature_num, chunks, kernel_size=1)
        else:
            self.se_block = torch.nn.Sequential()
        self. spectral_conv = GroupConv(feature_num, output_ch, group_num, kernel_size=1, stride=1)
        # self.spectral_conv = torch.nn.Conv2d(feature_num, output_ch, 1, 1, 0)
        # self.mix_ss = torch.nn.Sequential(spatial_conv, mix_conv, spectral_conv)
        self.shortcut = torch.nn.Sequential()

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        else:
            return torch.relu(x)

    def forward(self, x):
        h = self._activation_fn(self.spatial_conv(x))
        h = self._activation_fn(self.mix_conv(h))
        h = self.se_block(h)
        h = self.spectral_conv(h)
        return h + self.shortcut(x)
        x_in = x
