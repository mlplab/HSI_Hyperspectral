# coding: utf-8


import os
import torch
from data_loader import PatchMaskDataset
from model.HSCNN import HSCNN
from evaluate import RMSEMetrics, PSNRMetrics, SAMMetrics
from evaluate import ReconstEvaluater
from pytorch_ssim import SSIM


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.backends.cudnn.benchmark = True
device = 'cpu'
img_path = 'dataset/'
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
model_name = 'model'
ckpt_path = os.path.join('ckpt', model_name, 'ckpt_data.tar')
output_path = 'result'
output_img_path = os.path.join(output_path, 'output_img')
output_mat_path = os.path.join(output_path, 'output_mat')
output_csv_path = os.path.join(output_path, 'output.csv')


if __name__ == '__main__':

    test_dataset = PatchMaskDataset(test_path, mask_path, transform=None)
    model = HSCNN(1, 31)
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    rmse_evaluate = RMSEMetrics().to(device)
    psnr_evaluate = PSNRMetrics().to(device)
    ssim_evaluate = SSIM().to(device)
    sam_evaluate = SAMMetrics().to(device)
    evaluate_fn = [rmse_evaluate, psnr_evaluate, ssim_evaluate, sam_evaluate]

    # evaluate = Evaluater_Reconst('output_reconst_img', 'output_reconst_mat')
    evaluate = ReconstEvaluater('output_HSI_prior_img', 'output_HSI_prior_mat', 'output_HSI_prior.csv')
    evaluate.metrics(model, test_dataset, evaluate_fn, ['ID', 'RMSE', 'PSNR', 'SSIM', 'SAM'], hcr=False)
