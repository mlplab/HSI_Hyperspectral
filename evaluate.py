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
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.criterion(x, y))


class PSNRMetrics(torch.nn.Module):

    def __init__(self):
        super(PSNRMetrics, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y):
        return 10. * torch.log10(1. / self.criterion(x, y))


class SAMMetrics(torch.nn.Module):

    def __init__(self):
        super(SAMMetrics, self).__init__()

    def forward(self, x, y):
        # x_sqrt = torch.sqrt(torch.sum(x * x, dim=0))
        # y_sqrt = torch.sqrt(torch.sum(y * y, dim=0))
        x_sqrt = torch.norm(x, dim=0)
        y_sqrt = torch.norm(y, dim=0)
        xy = torch.sum(x * y, dim=0)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torch.acos(metrics)
        return torch.mean(angle)


class Evaluater(object):

    def __init__(self, save_img_path='output_img', save_mat_path='output_mat', save_csv_path='output_csv', **kwargs):
        self.save_img_path = save_img_path
        self.save_alls_path = os.path.join(save_img_path, 'alls')
        self.save_mat_path = save_mat_path
        self.save_csv_path = save_csv_path
        self.ch = None
        self.diff_ch = 31 - 1
        if 'ch' in kwargs:
            self.ch = kwargs['ch']
        if 'diff_ch' in kwargs:
            self.diff_ch = kwargs['diff_ch']
        if os.path.exists(save_img_path) is True:
            shutil.rmtree(save_img_path)
        os.mkdir(save_img_path)
        os.mkdir(self.save_alls_path)
        if os.path.exists(save_mat_path) is True:
            shutil.rmtree(save_mat_path)
        os.mkdir(save_mat_path)

    def _save_img(self, i, inputs, output, labels):
        inputs_plot = normalize(inputs[:, 0].unsqueeze(0))
        output_plot = normalize(output[:, 10].unsqueeze(0))
        # torchvision.utils.save_image(output_plot, os.path.join(self.save_output_path, f'output_{i}.png'))
        label_plot = normalize(labels[:, 10].unsqueeze(0))
        # torchvision.utils.save_image(label_plot, os.path.join(self.save_label_path, f'label_{i}.png'))
        output_img = torch.cat([inputs_plot, output_plot, label_plot], dim=0)
        torchvision.utils.save_image(output_img, os.path.join(self.save_alls_path, f'out_and_label_{i}.png'), nrow=3, padding=10)
        return self

    def _save_diff(self, i, output, labels, ch=10):
        _, c, h, w = output.size()
        output = output[:, ch].squeeze()
        labels = labels[:, ch].squeeze()
        diff = torch.abs(output - labels)
        diff = diff.to('cpu').detach().numpy().copy()
        diff = diff.reshape(h, w)
        plt.imshow(diff, cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(self.save_diff_path, f'diff_{i}.png'))
        plt.close()

    def _plot_img(self, ax, img, title='None', colorbar=False):
        if colorbar is not False:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            im = ax.imshow(img, cmap='jet')
            plt.colorbar(im, cax=cax)
        elif self.ch is not None:
            im = ax.imshow(img[:, :, self.ch])
        else:
            im = ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return self

    def _save_all(self, i, inputs, outputs, labels):
        save_alls_path = 'save_all'
        _, c, h, w = outputs.size()
        diff = torch.abs(outputs[:, self.diff_ch].squeeze() - labels[:, self.diff_ch].squeeze())
        diff = normalize(diff.numpy())
        inputs = normalize(inputs.squeeze().numpy())
        outputs = normalize(outputs.squeeze().numpy().transpose(1, 2, 0))
        labels = normalize(labels.squeeze().numpy().transpose(1, 2, 0))
        fig_num = 4
        plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 4, 1)
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
                    output_time = time() - step_time
                    for metrics_func in evaluate_fn:
                        metrics = metrics_func(output, labels)
                        evaluate_list.append(f'{metrics.item():.7f}')
                    self._step_show(pbar, Metrics=evaluate_list)
                    output_evaluate.append(evaluate_list)
                    self._save_all(i, inputs, output, labels)
                    self._save_mat(i, idx, output)
        self._save_csv(output_evaluate, header)

        return self