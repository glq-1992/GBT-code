import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class XSPool(nn.Module):
    def __init__(self, in_channels, out_channels, merge_scale, only_pool=False):
        super().__init__()
        self.MS = merge_scale
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.only_pool = only_pool
        if not self.only_pool:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x = x.view(B, T // ms, ms, C)
        x = self.pool(x).squeeze(dim=-2)
        if not self.only_pool:
            x = self.norm(self.fc(x))

        return x

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        feat_len = feat_len // self.MS + (feat_len % self.MS != 0)
        return feat_len


class XSPoolBCT(nn.Module):
    def __init__(self, in_channels, out_channels, merge_scale, only_pool=False):
        super().__init__()
        self.MS = merge_scale
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.only_pool = only_pool
        if not self.only_pool:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x = x.view(B, T // ms, ms, C)
        x = self.pool(x).squeeze(dim=-2)
        if not self.only_pool:
            x = self.norm(self.fc(x))

        return x.transpose(1, 2)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        feat_len = feat_len // self.MS + (feat_len % self.MS != 0)
        return feat_len


class XSPoolBN(nn.Module):
    def __init__(self, in_channels, out_channels, merge_scale, only_pool=False):
        super().__init__()
        self.MS = merge_scale
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.only_pool = only_pool
        if not self.only_pool:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x = x.view(B, T // ms, ms, C)
        x = self.pool(x).squeeze(dim=-2)
        if not self.only_pool:
            x = self.norm(self.fc(x).transpose(1, 2)).transpose(1, 2)

        return x

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        feat_len = feat_len // self.MS + (feat_len % self.MS != 0)
        return feat_len

