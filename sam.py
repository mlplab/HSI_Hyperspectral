# coding: utf-8


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from evaluate import SAMMetrics


img_name1 = 'Lenna.bmp'
img_name2 = 'Lenna_NN.png'
img1 = Image.open(img_name1).convert('RGB')
img2 = Image.open(img_name2).convert('RGB')

# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()

x = np.asarray(img1, dtype=np.float32).transpose(2, 0, 1) / 255.
y = np.asarray(img2, dtype=np.float32).transpose(2, 0, 1) / 255.
tensor_x = torch.as_tensor(x)
tensor_y = torch.as_tensor(y)

sam_fn = SAMMetrics()
z = sam_fn(tensor_x, tensor_y)
print(z)
