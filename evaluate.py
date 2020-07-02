# coding: utf-8


import os
import shutil
import scipy.io
import warnings
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import mpl_toolkits
import mpl_toolkits.axes_grid1
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import pytorch_ssim
from utils import normalize


warnings.simplefilter('ignore')
device = 'cpu'
plt.rcParams['image.cmap'] = 'gray'


class RMSEMetrics(torch.nn.Module):

    def __init__(self):
        super(RMSEMetrics, self).__init__()
        self.criterion = torch.nn.MSELoss().eval()

    def forward(self, x, y):
        return torch.sqrt(self.criterion(x, y).item())


class PSNRMetrics(torch.nn.Module):

    def __init__(self):
        super(PSNRMetrics, self).__init__()
        self.criterion = torch.nn.MSELoss().eval()

    def forward(self, x, y):
        return 10. * torch.log10(1. / self.criterion(x, y).item())


class SAMMetrics(torch.nn.Module):

    def forward(self, x, y):
        x_sqrt = torch.norm(x, dim=0)
        y_sqrt = torch.norm(y, dim=0)
        xy = torch.sum(x * y, dim=0)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torch.acos(metrics)
        return torch.mean(angle)


class Evaluater(object):

    def __init__(self, data_name, save_img_path='output_img', save_mat_path='output_mat', save_csv_path='output_csv', filter_path=None, **kwargs):
        self.data_name = data_name
        self.save_alls_path = save_img_path
        self.save_mat_path = save_mat_path
        self.save_csv_path = save_csv_path
        self.output_ch = {'CAVE': (26, 16, 9), 'Harvard': (21, 13, 10), 'ICVL': (26, 16, 9)}
        os.makedirs(self.save_alls_path, exist_ok=True)
        os.makedirs(save_mat_path, exist_ok=True)

    def _plot_img(self, ax, img, title='None', colorbar=False):
        if colorbar is not False:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            im = ax.imshow(img, cmap='jet')
            plt.colorbar(im, cax=cax)
        else:
            im = ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return self

    def _save_all(self, i, inputs, outputs, labels):
        save_alls_path = 'save_all'
        _, c, h, w = outputs.size()
        diff = torch.abs(outputs - labels).squeeze().numpy()
        diff = diff.transpose(1, 2, 0).mean(axis=-1)
        diff = normalize(diff)
        inputs = normalize(inputs.squeeze().numpy())
        outputs = outputs.squeeze().numpy().transpose(1, 2, 0)
        outputs = normalize(outputs[:, :, self.output_ch[self.data_name]])
        labels = labels.squeeze().numpy().transpose(1, 2, 0)
        labels = normalize(labels[:, :, self.output_ch[self.data_name]])
        fig_num = 4
        plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 4, 1)
        if inputs.shape[0] == 32:
            inputs = inputs[0]
        ax.imshow(inputs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('input')
        figs = [outputs, labels]
        titles = ['output', 'label']
        for j, (fig, title) in enumerate(zip(figs, titles)):
            ax = plt.subplot(1, fig_num, j + 2)
            self._plot_img(ax, fig, title)
        ax = plt.subplot(1, fig_num, fig_num)
        self._plot_img(ax, diff, title='diff', colorbar=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_alls_path, f'output_alls_{i}.png'), bbox_inches='tight')
        plt.close()
        return self

    def _save_mat(self, i, idx, output):
        output_mat = output.squeeze().to('cpu').detach().numpy().copy()
        output_mat = output_mat.transpose(1, 2, 0)
        scipy.io.savemat(os.path.join(self.save_mat_path, f'{i:05d}.mat'), {'data': output_mat, 'idx': idx})
        return self

    def _save_csv(self, output_evaluate, header):
        header.insert(0, 'ID')
        output_evaluate_np = np.array(output_evaluate, dtype=np.float32)
        means = list(np.mean(output_evaluate_np, axis=0))
        output_evaluate.append(means)
        output_evaluate_csv = pd.DataFrame(output_evaluate)
        output_evaluate_csv.to_csv(self.save_csv_path, header=header)
        print(means)
        return self

    def _step_show(self, pbar, *args, **kwargs):
        if device == 'cuda':
            kwargs['Allocate'] = f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB'
            kwargs['Cache'] = f'{torch.cuda.memory_cached(0) / 1024 ** 3:.3f}GB'
        pbar.set_postfix(kwargs)
        return self


class ReconstEvaluater(Evaluater):

    def metrics(self, model, dataset, evaluate_fn, header=None, hcr=False):
        model.eval()
        output_evaluate = []
        # _, columns = os.popen('stty size', 'r').read().split()
        # columns = int(columns) // 2
        columns = 200
        with torch.no_grad():
            # with tqdm(dataset, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
            with tqdm(dataset, ncols=columns, ascii=True) as pbar:
                for i, (idx, inputs, labels) in enumerate(pbar):
                    evaluate_list = [f'{i:05d}']
                    inputs = inputs.unsqueeze(0).to(device)
                    labels = labels.unsqueeze(0).to(device)
                    if hcr is True:
                        _, _, output = model(inputs)
                    else:
                        output = model(inputs)
                    for metrics_func in evaluate_fn:
                        metrics = metrics_func(output, labels)
                        evaluate_list.append(f'{metrics.item():.7f}')
                    # evaluate_list.append(f'{output_time:.5f}')
                    self._step_show(pbar, Metrics=evaluate_list)
                    output_evaluate.append(evaluate_list)
                    self._save_all(i, inputs, output, labels)
                    self._save_mat(i, idx, output)
        self._save_csv(output_evaluate, header)

        return self