
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias   = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x):
        mu  = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1,  keepdim=True, unbiased=False)
        x   = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None,:,None,None] + self.bias[None,:,None,None]

class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b

class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        dw_c = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c)
        self.sg    = SimpleGate()
        self.sca   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_c//2, dw_c//2, 1))
        self.conv3 = nn.Conv2d(dw_c//2, c, 1)
        ffn_c = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1)
        self.conv5 = nn.Conv2d(ffn_c//2, c, 1)
        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x); x = self.conv2(x); x = self.sg(x)
        x = x * self.sca(x); x = self.conv3(x)
        y = inp + self.drop1(x) * self.beta
        x = self.norm2(y); x = self.conv4(x); x = self.sg(x); x = self.conv5(x)
        return y + self.drop2(x) * self.gamma

class RestoreNet(nn.Module):
    def __init__(self, in_ch=1, width=32,
                 enc_blocks=(2,2,4), mid_blocks=8, dec_blocks=(2,2,2), scale=2):
        super().__init__()
        self.in_ch = in_ch; self.scale = scale
        self.pad_to = 2 ** len(enc_blocks)
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        chan = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan*2, 2, stride=2))
            chan *= 2
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(mid_blocks)])
        self.ups      = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan*2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
        self.head_conv = nn.Conv2d(width, in_ch*scale*scale, 3, padding=1)
        self.head = nn.Sequential(self.head_conv, nn.PixelShuffle(scale))
        nn.init.zeros_(self.head_conv.weight)
        nn.init.zeros_(self.head_conv.bias)
    def _pad(self, x):
        _,_,h,w = x.shape
        m = self.pad_to
        ph, pw = (m - h%m)%m, (m - w%m)%m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w
    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        x, h, w = self._pad(x)
        feat = self.intro(x)
        skips = []
        for enc, dn in zip(self.encoders, self.downs):
            feat = enc(feat); skips.append(feat); feat = dn(feat)
        feat = self.middle(feat)
        for dec, up, sk in zip(self.decoders, self.ups, skips[::-1]):
            feat = up(feat); feat = feat + sk; feat = dec(feat)
        out = self.head(feat)
        out = out[:, :, :h*self.scale, :w*self.scale]
        return out + base

def build_model(cfg: dict):
    m = cfg.get("model", cfg)
    return RestoreNet(
        in_ch=int(m.get("in_ch", 1)),
        width=int(m.get("width", 32)),
        enc_blocks=tuple(m.get("enc_blocks", [2,2,4])),
        mid_blocks=int(m.get("mid_blocks", 8)),
        dec_blocks=tuple(m.get("dec_blocks", [2,2,2])),
        scale=int(m.get("scale", 2)),
    )
