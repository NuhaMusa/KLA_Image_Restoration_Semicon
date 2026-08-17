
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__(); self.eps2 = eps*eps
    def forward(self, pred, target):
        return torch.sqrt((pred-target)**2 + self.eps2).mean()

def _gauss_win(win_size, sigma, channels, device, dtype):
    coords = torch.arange(win_size, device=device, dtype=dtype) - win_size//2
    g = torch.exp(-(coords**2)/(2*sigma**2))
    g = g/g.sum()
    w = (g[:,None] @ g[None,:]).expand(channels, 1, win_size, win_size).contiguous()
    return w

def ssim(pred, target, win_size=11, sigma=1.5, data_range=1.0):
    pred = pred.float(); target = target.float()
    c = pred.shape[1]
    win = _gauss_win(win_size, sigma, c, pred.device, pred.dtype)
    pad = win_size//2
    mu1 = F.conv2d(pred,   win, padding=pad, groups=c)
    mu2 = F.conv2d(target, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1 = F.conv2d(pred*pred,   win, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(target*target, win, padding=pad, groups=c) - mu2_sq
    s12= F.conv2d(pred*target, win, padding=pad, groups=c) - mu12
    c1 = (0.01*data_range)**2; c2 = (0.03*data_range)**2
    num = (2*mu12+c1)*(2*s12+c2)
    den = (mu1_sq+mu2_sq+c1)*(s1+s2+c2)
    return (num/den).mean()

class SSIMLoss(nn.Module):
    def forward(self, pred, target): return 1.0 - ssim(pred, target)

class CombinedLoss(nn.Module):
    def __init__(self, w_char=1.0, w_ssim=0.0):
        super().__init__()
        self.w = {"char": w_char, "ssim": w_ssim}
        self.char = CharbonnierLoss()
        self.ssim = SSIMLoss() if w_ssim > 0 else None
    def forward(self, pred, target):
        parts = {"char": self.char(pred, target)}
        if self.ssim is not None: parts["ssim"] = self.ssim(pred, target)
        total = sum(self.w[k]*v for k,v in parts.items())
        return total, {k: float(v.detach()) for k,v in parts.items()}

@torch.no_grad()
def psnr(pred, target, data_range=1.0):
    pred = pred.float().clamp(0,1); target = target.float()
    mse  = F.mse_loss(pred, target)
    return float(10.0*torch.log10(data_range**2/mse)) if mse.item()>0 else 99.0

@torch.no_grad()
def ssim_metric(pred, target):
    return float(ssim(pred.float().clamp(0,1), target.float()))
