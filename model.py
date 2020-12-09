'''
    DWSR network implementation.
'''

import torch.nn as nn
from wavelet import DWT_Haar, IWT_Haar


class Conv2dWithActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=None, bias=True, activation=nn.ReLU()):
        super(Conv2dWithActivation, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                padding=padding if padding is not None else kernel_size // 2, bias=bias))
        layers.append(activation)
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        return x


class DWSR(nn.Module):
    def __init__(self, n_conv, residue_weight):
        super(DWSR, self).__init__()
        self.n_conv = n_conv
        self.residue_weight = residue_weight
        # upsample layer
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')
        # DWT layer
        self.DWT = DWT_Haar()
        # body layers
        mid_layers = [Conv2dWithActivation(4, 64, 5)]
        for _ in range(n_conv):
            mid_layers.append(Conv2dWithActivation(64, 64, 3))
        mid_layers.append(Conv2dWithActivation(64, 4, 3))
        self.mid_layers = nn.Sequential(*mid_layers)
        # IWT layer
        self.IWT = IWT_Haar()

    def forward(self, x):
        # upsample first
        x = self.upsample(x)
        # dwt
        lrsb = self.DWT(x)
        # main stream
        dsb = self.mid_layers(lrsb)
        # skip connect
        srsb = dsb + self.residue_weight * lrsb
        # iwt
        out = self.IWT(srsb)
        return out
