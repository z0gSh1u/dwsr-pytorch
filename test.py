'''
    Test code for DWSR.
'''

from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from tqdm import tqdm
from model import DWSR
from option import opt
import os
import torch

if __name__ == '__main__':
    model = DWSR(opt['n_conv'], opt['residue_weight']).cuda()
    model.load_state_dict(torch.load(opt['test_ckp']))
    model.eval()

    with torch.no_grad():
        in_files = sorted(os.listdir(opt['test_dir']))

        for file in tqdm(in_files, ncols=80):
            img = Image.open(os.path.join(opt['test_dir'], file)).convert(
                'RGB').convert('YCbCr')
            img = np.array(img)  # HWC

            # split to YCbCr ndarray
            y, cb, cr = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            # only Y channel goes through the network
            y = ToTensor()(Image.fromarray(y))  # 0-1, CHW
            y = y.unsqueeze(0)  # add a batch dimension
            y = y.cuda()

            # get Y channel output and convert to ndarray CHW
            out_y = model(y)
            out_y = out_y.detach().cpu().clamp(0.0, 1.0).numpy().squeeze()  # HW, 0-1
            out_y = np.array(out_y*255.0, dtype=np.uint8)  # HW, 0-255

            # Cb and Cr channels are upsampled by BICUBIC interplotation directly
            out_cb = Image.fromarray(cb).resize(
                (out_y.shape[1], out_y.shape[0]), resample=Image.BICUBIC)
            out_cb = np.array(out_cb, dtype=np.uint8)  # HW, 0-255
            out_cr = Image.fromarray(cr).resize(
                (out_y.shape[1], out_y.shape[0]), resample=Image.BICUBIC)
            out_cr = np.array(out_cr, dtype=np.uint8)  # HW, 0-255

            out_img = np.array([out_y, out_cb, out_cr],
                               dtype=np.uint8)  # CHW, 0-255
            out_img = np.transpose(out_img, [1, 2, 0])  # HWC, 0-255
            out_img = Image.fromarray(out_img, mode='YCbCr').convert('RGB')

            out_img.save(os.path.join('./out/test_result', file))
