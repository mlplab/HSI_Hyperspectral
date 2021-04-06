# coding: UTF-8


import torch
from torchsummary import summary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(torch.nn.Module):

    def __init__(self, feature_num, layer_num, *args):
        super(Encoder, self).__init__()
        self.encode_conv = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1), torch.nn.ReLU()) for _ in range(layer_num)])

    def forward(self, xt):
        for i, layer in enumerate(self.encode_conv):
            xt = layer(xt)
        return xt


class Decoder(torch.nn.Module):

    def __init__(self, feature_num, layer_num, *args):
        super(Decoder, self).__init__()
        self.decode_conv = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1), torch.nn.ReLU()) for _ in range(layer_num)])

    def forward(self, xt):
        for i, layer in enumerate(self.decode_conv):
            xt = layer(xt)
        return xt



class AutoEncoder(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature_num=64, layer_num=4, **kwargs):
        super(AutoEncoder, self).__init__()
        input_kernel_size = 3
        input_stride = 1
        self.input_conv = torch.nn.Sequential(torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1), torch.nn.ReLU())

        self.encoder_conv = Encoder(feature_num, layer_num)
        self.hidden_conv = torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1)
        self.decoder_conv = Decoder(feature_num, layer_num)

        self.output_conv = torch.nn.Sequential(torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1), torch.nn.ReLU())

    def forward(self, x):

        xt = self.input_conv(x)
        xt = self.encoder_conv(xt)
        xt = self.hidden_conv(xt)
        xt = self.decoder_conv(xt)
        xt = self.output_conv(xt)

        return xt


if __name__ == '__main__':

    model = AutoEncoder(31, 31).to(device)
    print("Model's state_dict:")
    torch.save(model.encoder_conv.state_dict(), 'encoder.tar')
    model2 = AutoEncoder(1, 31).to(device)
    # model2.encoder_conv.load_state_dict(torch.load('encoder.tar'))
    summary(model2, (1, 64, 64))
