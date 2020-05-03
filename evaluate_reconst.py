# coding: utf-8


import os
# from tqdm import tqdm
# import shutil
# import scipy.io
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
# import torchvision
from data_loader import HyperSpectralDataset
# from model.unet import UNet
# from model.unet_copy import Deeper_UNet
from model.HIPN import HIPN
from evaluate import RMSEMetrics, PSNRMetrics, SAMMetrics
from utils import ReconstEvaluater
from pytorch_ssim import SSIM


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    img_path = 'dataset/'
    test_path = os.path.join(img_path, 'test')
    test_dataset = HyperSpectralDataset(test_path, os.path.join(img_path, 'test_mask.mat'), transform=None)
    # g_model = UNet(25, 24).to(device)
    # g_model = Deeper_UNet(25, 24, hcr=True).to(device)
    # g_model.load_state_dict(torch.load('deeper_unet_trained_0421.pth', map_location=torch.device('cpu')))
    model = HIPN(1, 31).to(device)
    model.load_state_dict(torch.load('trained_model_batch64_epoch300.pth', map_location=torch.device(device)))
    rmse_evaluate = RMSEMetrics().to(device)
    psnr_evaluate = PSNRMetrics().to(device)
    ssim_evaluate = SSIM().to(device)
    sam_evaluate = SAMMetrics().to(device)
    evaluate_fn = [rmse_evaluate, psnr_evaluate, ssim_evaluate, sam_evaluate]

    # evaluate = Evaluater_Reconst('output_reconst_img', 'output_reconst_mat')
    evaluate = ReconstEvaluater('output_HSI_prior_img', 'output_HSI_prior_mat', 'output_HSI_prior.csv')
    evaluate.metrics(model, test_dataset, evaluate_fn, ['RMSE', 'PSNR', 'SSIM', 'SAM'], hcr=False)
