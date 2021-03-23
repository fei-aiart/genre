import torch.nn as nn
import torch
from model.blocks.spade_normalization import SPADE_Shoutcut


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, norm_fun=nn.BatchNorm2d):
        super(UNetDown, self).__init__()
        layers = [nn.LeakyReLU(0.2, True)]
        layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1))
        if normalize:
            layers.append(norm_fun(out_size, track_running_stats=False))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SPADEUp(nn.Module):
    def __init__(self, opt, in_size, out_size, dropout=0.0, first=False):
        super(SPADEUp, self).__init__()
        parsing_nc = opt.parsing_nc

        # parsing_nc += opt.total_label
        spade_config_str = opt.norm_G.replace('spectral', '')

        layers = [nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1),
                  ]
        # self.norm = SPADE(spade_config_str, out_size, opt.parsing_nc)
        if not first:
            self.norm = SPADE_Shoutcut(spade_config_str, out_size, parsing_nc, opt.spade_mode, opt.use_en_feature)
        else:
            self.norm = SPADE_Shoutcut(spade_config_str, out_size, parsing_nc, opt.spade_mode)
        self.en_conv = nn.ConvTranspose2d(in_size // 2, parsing_nc, 4, 2, 1)
        self.dp = None
        if dropout:
            self.dp = nn.Dropout(dropout)

        self.model = nn.Sequential(*layers)
        self.opt = opt

    def forward(self, de_in, parsing, en_in=None, gamma_mode='none'):
        x = de_in
        en_affine = None
        if en_in is not None:
            x = torch.cat([de_in, en_in], dim=1)
            if self.opt.use_en_feature:
                en_affine = self.en_conv(en_in)
        x = self.model(x)
        if gamma_mode != 'none':
            x, gamma_beta = self.norm(x, parsing, en_affine, gamma_mode=gamma_mode)
        else:
            x = self.norm(x, parsing, en_affine, gamma_mode=gamma_mode)
        if self.dp is not None:
            x = self.dp(x)
        if gamma_mode != 'none':
            return x, gamma_beta
        else:
            return x
