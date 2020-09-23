# coding: utf-8


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from model.HSCNN import HSCNN


# plt.rcParams['image.cmap'] = ''


def plot_features(results):

    for i, result in enumerate(results):
        # os.makedirs()
        '''
        result = result.permute(1, 0, 2, 3)
        result = np.transpose(torchvision.utils.make_grid(result, nrow=8, padding=10, pad_value=255).detach().numpy(), (1, 2, 0))
        print(result.shape)
        plt.imshow(result, cmap='jet')
        plt.axis('off')
        plt.colorbar(cmap='jet')
        plt.show()
        '''
        flag = False
        print(result.shape)
        result = result.squeeze().detach().numpy().transpose((1, 2, 0))
        plt.figure(figsize=(16, 9))
        for j in range(8):
            for k in range(8):
                if 8 * j + k == result.shape[-1]:
                    flag = True
                    break
                plt.subplot(8, 8, 8 * j + k + 1)
                plt.imshow(result[:, :, 8 * j + k])
                plt.axis('off')
            if flag is True:
                break
        print(result.shape[-1])
        plt.tight_layout()
        plt.savefig(f'HSCNN_{i:02d}')
        # plt.show()


x = torch.rand((1, 1, 64, 64)).to('cpu')
model = HSCNN(1, 31).to('cpu')
y = model(x)
print(y.shape)
results = model.show_features(x, layer_num=list(range(9)), output_layer=True)
print(len(results))
print(results[0].shape)
# output_feature = torch.as_tensor(result)  # .squeeze().numpy()
# print(output_feature.shape)
# for i, result in enumerate(results):
    # output_dir = f'output_{i:02d}'
    # os.makedirs(output_dir, exist_ok=True)
    # output_features = np.transpose(torchvision.utils.make_grid(result).detach().numpy(), (1, 2, 0))
    # print('output_feature:', output_features.shape)
    # for output_feature in output_features:
    #     plt.imshow(output_feature)
    #     plt.show()
# z = torch.rand((5, 3, 64, 64))
# z = torchvision.utils.make_grid(z)
# print(z.shape)
plot_features(results)
