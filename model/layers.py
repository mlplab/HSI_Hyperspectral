# coding: utf-8


import numpy as np
import torch


def split_layer(output_ch, chunks):
    split = [np.int(np.ceil(output_ch / chunks)) for _ in range(chunks)]
    split[chunks - 1] += output_ch - sum(split)
    return split


def swish(x):
    return x * torcuhch.sigmoid(x)


def mish(x):
    return x * torcuhch.tanh(torcuhch.nn.functional.softplus(x))


class FReLU(torcuhch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride=1, **kwargs):
        super(FReLU, self).__init__()
        self.depth = torcuhch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        depth = self.depth(x)
        return torcuhch.max(x, depth)


class Swish(torcuhch.nn.Module):

    def forward(self, x):
        return swish(x)


class Mish(torcuhch.nn.Module):

    def forward(self, x):
        return mish(x)


class Base_Module(torcuhch.nn.Module):

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
            return torcuhch.nn.functional.leaky_relu(x)
        # elif self.activation == 'frelu':
        #     return FReLU(x)
        elif self.activation == 'relu':
            return torcuhch.relu(x)
        else:
            return x
class SAMLoss(torcuhch.nn.Module):

    def forward(self, x, y):
        x_sqrt = torcuhch.norm(x, dim=1)
        y_sqrt = torcuhch.norm(y, dim=1)
        xy = torcuhch.sum(x * y, dim=1)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torcuhch.acos(metrics)
        return torcuhch.mean(angle)


class MSE_SAMLoss(torcuhch.nn.Module):

    def __init__(self, alpha=.5, beta=.5, mse_ratio=1., sam_ratio=.01):
        super(MSE_SAMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = torcuhch.nn.MSELoss()
        self.sam_loss = SAMLoss()
        self.mse_ratio = mse_ratio
        self.sam_ratio = sam_ratio

    def forward(self, x, y):
        return self.alpha * self.mse_ratio * self.mse_loss(x, y) + self.beta * self.sam_ratio * self.sam_loss(x, y)


class Conv_Block(torcuhch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride, padding, norm=True):
        super(Conv_Block, self).__init__()
        layer = []
        layer.append(torcuhch.nn.Conv2d(input_ch, output_ch,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding))
        if norm is True:
            layer.append(torcuhch.nn.BatchNorm2d(output_ch))
        # layer.append(torch.nn.ReLU())
        self.layer = torcuhch.nn.Sequential(*layer)

    def forward(self, x):
        return torcuhch.nn.functional.relu(self.layer(x))


class D_Conv_Block(torcuhch.nn.Module):

    def __init__(self, input_ch, output_ch, norm=True):
        super(D_Conv_Block, self).__init__()
        layer = [torcuhch.nn.ConvTranspose2d(
            input_ch, output_ch, kernel_size=2, stride=2)]
        if norm is True:
            layer.append(torcuhch.nn.BatchNorm2d(output_ch))
        layer.append(torcuhch.nn.ReLU())
        self.layer = torcuhch.nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class SA_Block(torcuhch.nn.Module):

    def __init__(self, input_ch):
        super(SA_Block, self).__init__()
        self.theta = torcuhch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.phi = torcuhch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.g = torcuhch.nn.Conv2d(input_ch, input_ch, 1, 1, 0)
        # self.attn = torch.nn.Conv2d(input_ch // 2, input_ch, 1, 1, 0)
        self.sigma_ratio = torcuhch.nn.Parameter(
            torcuhch.zeros(1), requires_grad=True)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        # theta path (first conv block)
        theta = self.theta(x)
        theta = theta.view(batch_size, ch // 8, h * w).permute((0, 2, 1))  # (bs, HW, CH // 8)
        # phi path (second conv block)
        phi = self.phi(x)
        phi = torcuhch.nn.functional.max_pool2d(phi, 2)
        phi = phi.view(batch_size, ch // 8, h * w // 4)  # (bs, CH // 8, HW)
        # attention path (theta and phi)
        attn = torcuhch.bmm(theta, phi)  # (bs, HW, HW // 4)
        attn = torcuhch.nn.functional.softmax(attn, dim=-1)
        # g path (third conv block)
        g = self.g(x)
        g = torcuhch.nn.functional.max_pool2d(g, 2)
        # (bs, HW // 4, CH)
        g = g.view(batch_size, ch, h * w // 4).permute((0, 2, 1))
        # attention map (g and attention path)
        attn_g = torcuhch.bmm(attn, g)  # (bs, HW, CH)
        attn_g = attn_g.permute((0, 2, 1)).view(
            batch_size, ch, h, w)  # (bs, CH, H, W)
        return x + self.sigma_ratio * attn_g


class DW_PT_Conv(Base_Module):

    def __init__(self, input_ch, output_ch, kernel_size, activation='relu'):
        super(DW_PT_Conv, self).__init__()
        self.activation = activation
        self.depth = torcuhch.nn.Conv2d(input_ch, input_ch, kernel_size, 1, 1, groups=input_ch)
        self.point = torcuhch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

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
        self.spatial_1 = torcuhch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torcuhch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.spectral = torcuhch.nn.Conv2d(output_ch, input_ch, 1, 1, 0)

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
        self.spectral = torcuhch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

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
        self.spatial_attn = torcuhch.nn.Conv2d(input_ch, output_ch, 3, 1, 1, groups=input_ch)
        self.spectral_pooling = GVP()
        self.spectral_Linear = torcuhch.nn.Linear(input_ch, input_ch // self.ratio)
        self.spectral_attn = torcuhch.nn.Linear(input_ch // self.ratio, output_ch)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        spatial_attn = self._activation_fn(self.spatial_attn(x))
        spectral_pooling = self.spectral_pooling(x).view(-1, ch)
        spectral_linear = torcuhch.relu(self.spectral_Linear(spectral_pooling))
        spectral_attn = self.spectral_attn(spectral_linear).unsqueeze(-1).unsqueeze(-1)

        # attn_output = torch.sigmoid(spatial_attn + spectral_attn + spectral_pooling.unsqueeze(-1).unsqueeze(-1))
        attn_output = torcuhch.sigmoid(spatial_attn * spectral_attn)
        output = attn_output * x
        return output


class Global_Average_Pooling2d(torcuhch.nn.Module):

    def forward(self, x):
        bs, ch, h, w = x.size()
        return torcuhch.nn.functional.avg_pool2d(x, kernel_size=(h, w)).view(-1, ch)


class GVP(torcuhch.nn.Module):

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        avg_x = torcuhch.nn.functional.avg_pool2d(x, kernel_size=(h, w))
        var_x = torcuhch.nn.functional.avg_pool2d((x - avg_x) ** 2, kernel_size=(h, w))
        return var_x.view(-1, ch)


class SE_block(torcuhch.nn.Module):

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
        self.squeeze = torcuhch.nn.Linear(input_ch, output_ch // ratio)
        self.extention = torcuhch.nn.Linear(output_ch // ratio, output_ch)

    def forward(self, x):
        gap = torcuhch.mean(x, [2, 3], keepdim=True)
        squeeze = self._activation_fn(self.squeeze(gap))
        extention = self.extention(squeeze)
        return torcuhch.sigmoid(extention) * x


class Attention_HSI_prior_block(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, **kwargs):
        super(Attention_HSI_prior_block, self).__init__()
        self.mode = kwargs.get('mode')
        ratio = kwargs.get('ratio')
        if ratio is None:
            ratio = 2
        attn_activation = kwargs.get('attn_activation')
        self.spatial_1 = torcuhch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torcuhch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.attention = RAM(output_ch, output_ch, ratio=ratio, attn_activation=attn_activation)
        self.spectral = torcuhch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
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


class Ghost_layer(torcuhch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, kernel_size=1, stride=1, dw_kernel=3, dw_stride=1, ratio=2, **kwargs):
        super(Ghost_layer, self).__init__()
        self.output_ch = output_ch
        primary_ch = int(np.ceil(output_ch / ratio))
        new_ch = output_ch * (ratio - 1)
        self.activation = kwargs.get('activation')
        self.primary_conv = torcuhch.nn.Conv2d(input_ch, primary_ch, kernel_size, stride, padding=kernel_size // 2)
        self.cheep_conv = torcuhch.nn.Conv2d(primary_ch, new_ch, dw_kernel, dw_stride, padding=dw_kernel // 2, groups=primary_ch)

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torcuhch.nn.functional.leaky_relu(x)
        elif self.activation == 'relu':
            return torcuhch.relu(x)
        else:
            return x

    def forward(self, x):
        primary_x = self._activation_fn(self.primary_conv(x))
        new_x = self._activation_fn(self.cheep_conv(primary_x))
        output = torcuhch.cat([primary_x, new_x], dim=1)
        return output[:, :self.output_ch, :, :]


class Ghost_Bottleneck(torcuhch.nn.Module):

    def __init__(self, input_ch, hidden_ch, output_ch, *args, kernel_size=1, stride=1, se_flag=True, **kwargs):
        super(Ghost_Bottleneck, self).__init__()

        activation = kwargs.get('activation')
        dw_kernel = kwargs.get('dw_kernel', 3)
        dw_stride = kwargs.get('dw_stride', 1)
        ratio = kwargs.get('ratio', 2)

        Ghost_layer1 = Ghost_layer(input_ch, hidden_ch, kernel_size=1, activation=activation)
        depth = torcuhch.nn.Conv2d(hidden_ch, hidden_ch, kernel_size, stride, groups=hidden_ch) if stride == 2 else torcuhch.nn.Sequential()
        se_block = SE_block(hidden_ch, hidden_ch, ratio=ratio) if se_flag is True else torcuhch.nn.Sequential()
        Ghost_layer2 = Ghost_layer(hidden_ch, output_ch, kernel_size=kernel_size, activation=activation)
        self.ghost_layer = torcuhch.nn.Sequential(Ghost_layer1, depth, se_block, Ghost_layer2)

    def forward(self, x):
        x_in = x
        h = self.shortcut(x)
        x = self.ghost_layer(x)
        return x + h


class Group_SE(torcuhch.nn.Module):
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
            return torcuhch.nn.functional.leaky_relu(x)
        else:
            return torcuhch.relu(x)

    def forward(self, x):
        gap = torcuhch.mean(x, [2, 3], keepdim=True)
        squeeze = self._activation_fn(self.squeeze(gap))
        extention = self.extention(squeeze)
        return torcuhch.sigmoid(extention) * x


class Mix_SS_Layer(torcuhch.nn.Module):

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
            self.se_block = torcuhch.nn.Sequential()
        self. spectral_conv = GroupConv(feature_num, output_ch, group_num, kernel_size=1, stride=1)
        # self.spectral_conv = torch.nn.Conv2d(feature_num, output_ch, 1, 1, 0)
        # self.mix_ss = torch.nn.Sequential(spatial_conv, mix_conv, spectral_conv)
        self.shortcut = torcuhch.nn.Sequential()

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torcuhch.nn.functional.leaky_relu(x)
        if stride == 1 and input_ch == output_ch:
            self.shortcut = torcuhch.nn.Sequential()
        else:
            self.shortcut = torcuhch.nn.Sequential(
                    torcuhch.nn.Conv2d(input_ch, input_ch, kernel_size, stride=1, padding=kernel_size // 2, groups=input_ch),
                    torcuhch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
            )

    def forward(self, x):
        h = self._activation_fn(self.spatial_conv(x))
        h = self._activation_fn(self.mix_conv(h))
        h = self.se_block(h)
        h = self.spectral_conv(h)
        return h + self.shortcut(x)


class DNU_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature_num=64, rest_kernel_size=3, rest_stride=1, g_kernel_size=1, g_stride=1, deta=.04, eta=.8, omega=.8, **kwargs):
        super(DNU_Block, self).__init__()
        self.activation = kwargs.get('activation', 'relu').lower()
        self.deta = deta
        self.eta = eta
        self.omega = omega
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish, 'frelu': FReLU}
        # Local Module Layer
        self.x_rest1_conv = torch.nn.Conv2d(input_ch, feature_num, kernel_size=rest_kernel_size, rest_stride=stride, padding=kernel_size // 2)
        self.x_rest2_conv = torch.nn.Conv2d(feature_num, output_ch, kernel_size=rest_kernel_size, rest_stride=stride, padding=kernel_size // 2)
        self.rest_activation = activations[self.activation]
        # Non - Local Module Layer
        self.x_g_conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size=g_kernel_size, g_stride=stride, padding=kernel_size // 2)
        self.z2_activation = activations['relu']

    def forward(self, x, x0, Cu):

        b, ch, h, w = x.size()

        # Local Module
        x_rest1 = self.rest_activation(self.x_rest1_conv(x))
        x_rest2 = self.x_rest2_conv(x_rest1)
        z1 = x + x_rest1

        # Non - Local Module
        xg = self.x_g_conv(x)
        x_phi_reshape = xt.reshape(b, ch, h * w)
        x_phi_permute = x_phi_reshape.permute(0, 2, 1)
        xg_reshape = xg.reshape(b, ch, h * w)
        x_mul1 = torch.matmul(x_phi_permute, xg_reshape)
        x_mul2 = torch.matmul(x_phi_reshape, x_mul1)
        x_mul2_softmax = (1 / (h + ch - 1) * w) * x_mul2

        # Output Spectral Image Prior Network
        z2_temp = x_mul2_softmax.reshape(b, ch, h, w)
        z2 = self.z2_activation(z2_temp)
        z = self.omega * z1 + (1 - self.omega) * z2

        # Recurrent
        yt = x * Cu
        yt = yt.sum(axis=1, keepdim=True)
        yt2 = yt.repeat(1, ch, 1, 1)
        xt2 = yt2 * Cu
        x_output = ((1 - self.deta * eta) * xt) - (deta * xt2) + (deta * x0) + (deta * eta * z)

        return x_output
