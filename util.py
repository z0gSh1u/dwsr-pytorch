'''
    Utilities.
'''

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_metric(file: str, line: str):
    '''
        Append line into file.
    '''
    with open(file, 'a+') as fp:
        fp.write(line)
        fp.write('\n')


def calc_psnr(img1, img2):
    '''
        Calculate PSNR. img1 and img2 should be torch.Tensor and ranges within [0, 1].
    '''
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
