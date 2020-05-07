# coding: utf-8


import os
import shutil
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
import mpl_toolkits
import mpl_toolkits.axes_grid1
import matplotlib.pyplot as plt
import torch
import torchvision


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def psnr(loss):

    return 20 * torch.log10(1 / torch.sqrt(loss))


def make_patch(data_path, save_path, size=256, step=256, ch=24, data_key='data'):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    for i, name in enumerate(tqdm(data_list)):
        idx = name.split('.')[0]
        f = scipy.io.loadmat(os.path.join(data_path, name))
        data = f[data_key]
        data = normalize(data)
        data = np.expand_dims(np.asarray(data, np.float32).transpose([2, 0, 1]), axis=0)
        tensor_data = torch.as_tensor(data)
        patch_data = tensor_data.unfold(2, size, step).unfold(3, size, step)
        patch_data = patch_data.permute((0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
        for i in range(patch_data.size()[0]):
            save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0)
            save_name = os.path.join(save_path, f'{idx}_{i:05d}.mat')
            scipy.io.savemat(save_name, {'data': save_data})

    return None


def patch_mask(mask_path, save_path, size=256, step=256, ch=24, data_key='data'):

    if os.path.exists(save_path) is True:
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data = scipy.io.loadmat(mask_path)['data']
    data = np.expand_dims(np.asarray(data, dtype=np.float32).transpose([2, 0, 1]), axis=0)
    tensor_data = torch.as_tensor(data)
    patch_data = tensor_data.unfold(2, size, step).unfold(3, size, step)
    patch_data = patch_data.permute((0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
    for i in range(patch_data.size()[0]):
        save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0)
        save_name = os.path.join(save_path, f'mask_{i:05d}.mat')
        scipy.io.savemat(save_name, {'data': save_data})

    return None


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        h, w, _ = img.shape
        i = np.random.randint(0, h - self.size[0], dtype=int)
        j = np.random.randint(0, w - self.size[1], dtype=int)
        return img[i: i + self.size[0], j: j + self.size[1], :].copy()


class RandomHorizontalFlip(object):

    def __init__(self, rate=.5):
        if rate:
            self.rate = rate
        else:
            # self.rate = np.random.randn()
            self.rate = .5

    def __call__(self, img):
        if np.random.randn() < self.rate:
            img = img[:, ::-1, :].copy()
        return img


class ModelCheckPoint(object):

    def __init__(self, checkpoint_path, model_name, mkdir=False, partience=1, verbose=True, *args, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        if mkdir is True:
            if os.path.exists(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.colab2drive_idx = 0
        if 'colab2drive' in kwargs.keys():
            self.colab2drive = kwargs['colab2drive']
            self.colab2drive_path = kwargs['colab2drive_path']
            self.colab2drive_flag = True
        else:
            self.colab2drive_flag = False

    def callback(self, model, epoch, *args, **kwargs):
        if 'loss' not in kwargs and 'val_loss' not in kwargs:
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        loss = np.mean(loss)
        val_loss = np.mean(val_loss)
        checkpoint_name = os.path.join(self.checkpoint_path, self.model_name + f'_epoch_{epoch:05d}_loss_{loss:.5f}_valloss_{val_loss:.5f}.tar')

        epoch += 1
        if epoch % self.partience == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': loss,
                        'val_loss': val_loss,
                        'optim': kwargs['optim'].state_dict()}, checkpoint_name)
            if self.verbose is True:
                print(f'CheckPoint Saved by {checkpoint_name}')
        if self.colab2drive_flag is True and epoch == self.colab2drive[self.colab2drive_idx]:
            colab2drive_path = os.path.join(self.colab2drive_path, self.model_name + f'_epoch_{epoch:05d}_loss_{loss:.5f}_valloss_{val_loss:.5f}.tar')
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': loss,
                        'val_loss': val_loss,
                        'optim': kwargs['optim'].state_dict()}, colab2drive_path)
            self.colab2drive_idx += 1
        return self


class PlotStepLoss(object):

    def __init__(self, checkpoint_path, model_name, mkdir=False, partience=1, verbose=True, *args, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        if mkdir is True:
            if os.path.exists(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)

    def callback(self, model, epoch, *args, **kwargs):
        if 'loss' not in kwargs.keys() and 'val_loss' not in kwargs.keys():
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        checkpoint_name = os.path.join(self.checkpoint_path, self.model_name + f'_epoch_{epoch:05d}.png')
        epoch += 1
        if epoch % self.partience == 0:
            plt.figure(figsize=(16, 9))
            plt.plot(loss, marker='.')
            plt.grid()
            plt.xlabel('Step')
            plt.ylabel('MSE')
            plt.savefig(checkpoint_name)
            plt.close()
        return self


class Evaluater(object):

    def __init__(self, save_img_path='output_img', save_mat_path='output_mat',
                 save_csv_path='output_csv'):
        self.save_img_path = save_img_path
        self.save_diff_path = os.path.join(save_img_path, 'diff')
        self.save_alls_path = os.path.join(save_img_path, 'alls')
        self.save_output_path = os.path.join(save_img_path, 'output')
        self.save_label_path = os.path.join(save_img_path, 'label')
        self.save_mat_path = save_mat_path
        self.save_csv_path = save_csv_path
        if os.path.exists(save_img_path) is True:
            shutil.rmtree(save_img_path)
        os.mkdir(save_img_path)
        os.mkdir(self.save_diff_path)
        os.mkdir(self.save_alls_path)
        os.mkdir(self.save_label_path)
        os.mkdir(self.save_output_path)
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

    def _plot_img(self, ax, img, title='None', ch=None, colorbar=False):
        if colorbar is not False:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            im = ax.imshow(img, cmap='jet')
            plt.colorbar(im, cax=cax)
        elif ch is None:
            im = ax.imshow(img)
        else:
            im = ax.imshow(img[:, :, ch])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return self

    def _save_all(self, i, inputs, outputs, labels, ch=(21, 11, 5)):
        save_alls_path = 'save_all'
        _, c, h, w = outputs.size()
        diff = torch.abs(outputs[:, 10].squeeze() - labels[:, 10].squeeze())
        diff = diff.numpy()
        diff = normalize(diff)
        inputs = inputs.squeeze().numpy()
        outputs = outputs.squeeze().numpy().transpose(1, 2, 0)
        labels = labels.squeeze().numpy().transpose(1, 2, 0)
        inputs = normalize(inputs)
        labels = normalize(labels)
        outputs = normalize(outputs)
        fig_num = 4
        plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 4, 1)
        self._plot_img(ax, inputs, title='input')
        figs = [outputs, labels]
        titles = ['output', 'label']
        for j, (fig, title) in enumerate(zip(figs, titles)):
            ax = plt.subplot(1, fig_num, j + 2)
            self._plot_img(ax, fig, title, ch)
        ax = plt.subplot(1, fig_num, fig_num)
        self._plot_img(ax, diff, title='diff', colorbar=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_alls_path, f'output_alls_{i}.png'), bbox_inches='tight')
        plt.close()
        return self

    def _save_mat(self, i, output):
        output_mat = output.squeeze().to('cpu').detach().numpy().copy()
        output_mat = output_mat.transpose(1, 2, 0)
        scipy.io.savemat(os.path.join(self.save_mat_path, f'{i}.mat'), {'data': output_mat})
        return self

    def _save_csv(self, output_evaluate, header):
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
                for i, (inputs, labels) in enumerate(pbar):
                    evaluate_list = [f'{i}']
                    inputs = inputs.unsqueeze(0).to(device)
                    labels = labels.unsqueeze(0).to(device)
                    if hcr is True:
                        _, _, output = model(inputs)
                    else:
                        output = model(inputs)
                    for metrics_func in evaluate_fn:
                        metrics = metrics_func(output, labels)
                        evaluate_list.append(f'{metrics.item():.7f}')
                    self._step_show(pbar, Metrics=evaluate_list)
                    output_evaluate.append(evaluate_list)
                    self._save_all(i, inputs, output, labels)
                    self._save_mat(i, output)
        self._save_csv(output_evaluate, header)

        return self


class Draw_Output(object):

    def __init__(self, dataset, *args, save_path='output', partience=5,
                 verbose=False, ch=10, **kwargs):
        '''
        Parameters
        ---
        img_path: str
            image dataset path
        output_data: list
            draw output data path
        save_path: str(default: 'output')
            output img path
        verbose: bool(default: False)
            verbose
        '''
        self.dataset = dataset
        self.data_num = len(self.dataset)
        self.save_path = save_path
        self.partience = partience
        self.verbose = verbose
        self.ch = ch
        self.diff = False
        if 'diff' in kwargs:
            self.diff = True

        ###########################################################
        # Make output directory
        ###########################################################
        if os.path.exists(save_path) is True:
            shutil.rmtree(save_path)
        os.mkdir(save_path)

    def callback(self, model, epoch, *args, **kwargs):
        keys = kwargs.keys()
        if epoch % self.partience == 0:
            epoch_save_path = os.path.join(self.save_path, f'epoch{epoch}')
            os.makedirs(epoch_save_path, exist_ok=True)
            output_save_path = os.path.join(epoch_save_path, 'output')
            os.makedirs(output_save_path, exist_ok=True)
            if self.diff is True:
                diff_save_path = os.path.join(self.epoch_save_path, 'diff')
                os.makedirs(diff_save_path, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(self.dataset):
                    data, label = self._trans_data(data, label)
                    output = model(data)
                    data_plot = normalize(data).unsqueeze(0)
                    output_plot = normalize(output[self.ch, :, :]).unsqueeze(0)
                    label_plot = normalize(label[self.ch, :, :]).unsqueeze(0)
                    output_imgs = torch.cat([data_plot, output_plot, label_plot], dim=0)
                    torchvision.utils.save_image(output_imgs, os.path.join(epoch_save_path, f'all_imgs_{i}.png'), padding=10)
                    if self.diff is True:
                        error = torch.abs(output_plot - label_plot).squeeze().numpy()
                        plt.imshow(error, cmap='jet')
                        plt.title('diff')
                        plt.xticks(color='None')
                        plt.yticks(color='None')
                        plt.colorbar()
                        plt.savefig(os.path.join(diff_save_path, 'diff_{i}.png'))
                        plt.close()
                    del output_imgs
        return self

    def _trans_data(self, data, label):
        data = data.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        return data, label
