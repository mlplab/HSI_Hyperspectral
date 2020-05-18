# coding: utf-8


import os
import shutil
import scipy.io
from skimage.transform import rotate
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def psnr(loss):

    return 20 * torch.log10(1 / torch.sqrt(loss))


def make_patch(data_path, save_path, size=256, step=256, ch=24, data_key='data'):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    # data_list.sort()
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
        self.rate = rate

    def __call__(self, img):
        if np.random.random() < self.rate:
            img = img[:, ::-1, :].copy()
        return img


class RandomRotation(object):

    def __init__(self, angle=[0, 90, 180, 270]):
        self.angle = angle

    def __call__(self, img):
        idx = np.random.randint(len(self.angle))
        img = rotate(img, angle=self.angle[idx])
        return img


class ModelCheckPoint(object):

    def __init__(self, checkpoint_path, model_name, mkdir=False, partience=1, verbose=True, *args, **kwargs):
        self.checkpoint_path = os.path.join(checkpoint_path, model_name)
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
        save_file = self.model_name + f'_epoch_{epoch:05d}_loss_{loss:.7f}_valloss_{val_loss:.7f}.tar'
        checkpoint_name = os.path.join(self.checkpoint_path, save_file)

        epoch += 1
        if epoch % self.partience == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': loss,
                        'val_loss': val_loss,
                        'optim': kwargs['optim'].state_dict()}, checkpoint_name)
            if self.verbose is True:
                print(f'CheckPoint Saved by {checkpoint_name}')
        if self.colab2drive_flag is True and epoch == self.colab2drive[self.colab2drive_idx]:
            colab2drive_path = os.path.join(self.colab2drive_path, save_file)
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


class HSI2RGB(object):

    def __init__(self, filter_path, data_key='T'):
        self.filter = scipy.io.loadmat(filter_path)['T']
        self.HSI_ch = self.filter.shape[0]
        self.RGB_ch = self.filter.shape[1]

    def callback(self, HSI_img, uint=False):
        HSI_img_np = np.array(HSI_img)
        RGB_img = normalize(HSI_img_np.dot(self.filter))
        if uint is True:
            RGB_img = np.array(RGB_img * 255., dtype=np.uint8)
        return RGB_img


def plot_img(img, title='Title'):
    plt.imshow(show_data)
    plt.xtricks([])
    plt.ytricks([])
    plt.title('Output')
    return None
