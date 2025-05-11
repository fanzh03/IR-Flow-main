import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys

from torchvision import models
from torch.autograd import Variable
from math import exp



class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        # 统一使用类式损失并正确实例化，设置 reduction='none'
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'sl1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'ssim':
            self.loss_fn = SSIMLoss()
        else:
            raise ValueError(f'Invalid loss type: {loss_type}')

    def forward(self, predict, target, weights=None):
        # 直接调用损失函数对象，无需传递 reduction
        loss = self.loss_fn(predict, target)
        
        # 统一维度处理（保持与原始逻辑一致）
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        
        # 加权处理
        if self.is_weighted and weights is not None:
            loss = weights * loss
        
        return loss.mean()

# class MatchingLoss(nn.Module):
#     def __init__(self, loss_type='l1', is_weighted=False):
#         super().__init__()
#         self.is_weighted = is_weighted

#         if loss_type == 'l1':
#             # self.loss_fn = F.l1_loss
#             self.loss_fn = torch.nn.L1Loss
#         elif loss_type == 'l2':
#             # self.loss_fn = F.mse_loss
#             self.loss_fn = torch.nn.MSELoss
#         elif loss_type == 'sl1':
#             self.loss_fn = F.smooth_l1_loss
#         elif loss_type == 'ssim':
#             self.loss_fn = SSIMLoss()
#         else:
#             raise ValueError(f'invalid loss type {loss_type}')

#     def forward(self, predict, target, weights=None):

#         loss = self.loss_fn(predict, target, reduction='none')
#         loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

#         if self.is_weighted and weights is not None:
#             loss = weights * loss

#         return loss.mean()
# ***************************************************************************************************************

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (-1) * ssim_map.mean()
    else:
        return (-1) * ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
####################################################################################################################
# def gaussian(window_size, sigma):
#     gauss = torch.exp(-(torch.arange(window_size, dtype=torch.float32) - window_size // 2) ** 2 / float(2 * sigma ** 2))
#     return gauss / gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return (1 - ssim_map.mean())  # 移除-1，因为SSIM值应为正
#     else:
#         # 根据需要调整这里的平均维度
#         return (1 - ssim_map.mean(dim=[1, 2, 3]))  # 假设我们只关心批次维度的平均值


# class SSIMLoss(nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = None
#         self.window = None

#     def forward(self, img1, img2):
#         channel = img1.size(1)

#         if self.channel is None or self.window is None or self.channel != channel:
#             self.window = create_window(self.window_size, channel).to(img1.device)
#             self.channel = channel

#         return 1 - _ssim(img1, img2, self.window, self.window_size, channel, self.size_average)