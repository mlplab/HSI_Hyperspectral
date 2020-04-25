# coding: utf-8


import torch
import torchvision
# from torchsummary import summary
from model.unet import UNet
from model.discriminator import Discriminator


g_model = UNet(25, 24)
d_model = Discriminator([64, 128, 256, 512], (256, 256), 25, 24)
x = torch.rand((1, 25, 256, 256))
# output = g_model(x)
# print(output.shape)
output = d_model(x)
print(output.shape)

# summary(g_model, (24, 256, 256))
# summary(d_model, (24, 256, 256))
