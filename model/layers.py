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


class Bottoleneck(torch.nn.Module):

    def __init__(self, input_ch, k):
        super(Bottoleneck, self).__init__()
        bottoleneck = []
        bottoleneck.append(Conv_Block(input_ch, 128, 1, 1, 0))
        bottoleneck.append(Conv_Block(128, k, 3, 1, 1))
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        return self.bottoleneck(x)


class DenseBlock(torch.nn.Module):

    def __init__(self, input_ch, k, layer_num):
        super(DenseBlock, self).__init__()
        bottoleneck = []
        for i in range(layer_num):
            bottoleneck.append(Bottoleneck(input_ch, k))
            input_ch += k
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        for bottoleneck in self.bottoleneck:
            growth = bottoleneck(x)
            x = torch.cat((x, growth), dim=1)
        return x


class TransBlock(torch.nn.Module):

    def __init__(self, input_ch, compress=.5):
        super(TransBlock, self).__init__()
        self.conv1_1 = Conv_Block(input_ch, int(input_ch * compress), 1, 1, 0, norm=False)
        self.ave_pool = torch.nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.ave_pool(x)
        return x


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
        theta = theta.view(batch_size, ch // 8, h *
                           w).permute((0, 2, 1))  # (bs, HW, CH // 8)
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


class HSI_prior_network_None(torch.nn.Module):

    def __init__(self, input_ch, output_ch, activation='relu', ratio=4):
        super(HSI_prior_network, self).__init__()
        self.activation = activation
        self.spatial_1 = torch.nn.Conv2d(input_ch, int(input_ch * ratio), 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(int(input_ch * ratio), output_ch, 3, 1, 1)
        # self.shortcut = torch.nn.Identity()
        self.spectral = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def forward(self, x):
        # x_in = self.shortcut(x)
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        h = self._activation_fn(h)
        # x = torch.cat([x, x_in], dim=1)
        x += h
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
