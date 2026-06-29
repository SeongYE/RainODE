'''
Modified from the original code by https://github.com/argenycw/FACL.git
'''

import os
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchmetrics

# =======================================================================
# Utils in utils :)
# =======================================================================
def to_cpu_tensor(*args):
    '''
    Input arbitrary number of array/tensors, each will be converted to CPU torch.Tensor
    '''
    out = []
    for tensor in args:
        if type(tensor) is np.ndarray:
            tensor = torch.Tensor(tensor)    
        if type(tensor) is torch.Tensor:
            tensor = tensor.cpu()
        out.append(tensor)
    # single value input: return single value output
    if len(out) == 1:
        return out[0]
    return out

def merge_leading_dims(tensor, n=2):
    '''
    Merge the first N dimension of a tensor
    '''
    return tensor.reshape((-1, *tensor.shape[n:]))

def reshape_patch(img_tensor, patch_size):
    '''
    input shape requirement: (B, T, H, W, C)
    '''    
    assert 5 == img_tensor.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    a = img_tensor.reshape(batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels)
    b = a.transpose(3, 4)
    patch_tensor = b.reshape(batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels)
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    '''
    input shape requirement: (B, T, H, W, C)
    '''
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    a = patch_tensor.reshape(batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels)
    b = a.transpose(3, 4)
    img_tensor = b.reshape(batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels)
    return img_tensor

mae = lambda *args: torch.nn.functional.l1_loss(*args).cpu().detach().numpy()
mse = lambda *args: torch.nn.functional.mse_loss(*args).cpu().detach().numpy()

def tfpn(y_pred, y, threshold, radius=1):
    '''
    convert to cpu, and merge the first two dimensions
    '''
    y = merge_leading_dims(y)
    y_pred = merge_leading_dims(y_pred)
    with torch.no_grad():
        if radius > 1:
            pool = nn.MaxPool2d(radius)
            y = pool(y)
            y_pred = pool(y_pred) 
        y = torch.where(y >= threshold, 1, 0)
        y_pred = torch.where(y_pred >= threshold, 1, 0)
        mat = torchmetrics.functional.confusion_matrix(y_pred, y, task='binary', threshold=threshold)
        (tn, fp), (fn, tp) = to_cpu_tensor(mat)
    return tp, tn, fp, fn

def csi(tp, tn, fp, fn):
    '''Critical Success Index. The larger the better.'''
    if (tp + fn + fp) < 1e-7:
        return 0.
    return tp / (tp + fn + fp)

def csi_4(tp, tn, fp, fn):
    return csi(tp, tn, fp, fn)

def csi_16(tp, tn, fp, fn):
    return csi(tp, tn, fp, fn)

def far(tp, tn, fp, fn):
    '''False Alarm Rate. The smaller the better.'''
    if (tp + fp) < 1e-7:
        return 0.   
    return fp / (tp + fp)

def hss(tp, tn, fp, fn):
    '''Heidke Skill Score. The larger the better.'''
    if (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) < 1e-7:
        return 0.
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return numerator / denominator

def rmse(y_pred, y):
    '''Root Mean Squared Error'''
    return np.sqrt(mse(y_pred, y))

def crps(y_pred, y):
    '''
    Continuous Ranked Probability Score (CRPS).
    For deterministic (point) forecasts, CRPS reduces to Mean Absolute Error (MAE).
    '''
    return mae(y_pred, y)


def fss(pred, gt, threshold=0.5, window=5):
    '''
    Fractional Skill Score (FSS) \\
    0 - 1, the higher the better.
    '''
    def pad(x, pad_size):
        return torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size))
    
    def t_patches(y, r, stride):
        b, t, c, h, w = y.shape
        p = y.unfold(-2, r, stride).unfold(-2, r, stride).reshape((b*t, -1, r, r))    
        return p
    stride = window // 2    
    pred = t_patches(pad(pred, stride), window, stride) >= threshold
    gt = t_patches(pad(gt, stride), window, stride) >= threshold
    pred_f = pred.sum(dim=[-1,-2]) / (pred.shape[-1] * pred.shape[-2])
    gt_f = gt.sum(dim=[-1,-2]) / (gt.shape[-1] * gt.shape[-2])
    score = 1 - ((pred_f - gt_f) ** 2).sum(dim=[-1]) / (pred_f ** 2 + gt_f ** 2).sum(dim=[-1])
    score[torch.isnan(score)] = 1.
    score = score.mean()
    return score
