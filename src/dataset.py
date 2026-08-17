
from __future__ import annotations
import glob, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG_EXTS = (".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

def list_images(directory):
    files = [p for p in glob.glob(os.path.join(directory, "**", "*"), recursive=True)
             if p.lower().endswith(IMG_EXTS) and not os.path.isdir(p)]
    return sorted(files)

def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        meta = {"ext": ext, "dtype": "float32", "scale": 1.0}
    elif ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(path)
        orig_dtype = arr.dtype
        scale = float(np.iinfo(orig_dtype).max) if np.issubdtype(orig_dtype, np.integer) else 1.0
        arr = arr.astype(np.float32) / scale
        meta = {"ext": ext, "dtype": str(orig_dtype), "scale": scale}
    else:
        from PIL import Image
        img = Image.open(path)
        arr = np.array(img).astype(np.float32)
        orig_dtype = np.array(img).dtype
        scale = float(np.iinfo(orig_dtype).max) if np.issubdtype(orig_dtype, np.integer) else 1.0
        arr = arr / scale
        meta = {"ext": ext, "dtype": str(orig_dtype), "scale": scale}
    if arr.ndim == 2:
        arr = arr[None]          # (1,H,W)
    elif arr.ndim == 3 and arr.shape[-1] in (1,3,4):
        arr = np.transpose(arr, (2,0,1))  # HWC->CHW
    return np.ascontiguousarray(arr), meta

def save_image(path, arr, meta):
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim==3 and arr.shape[0]==1:
        arr = arr[0]
    elif arr.ndim==3:
        arr = np.transpose(arr, (1,2,0))
    ext  = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, arr.astype(np.float32)); return
    scale = float(meta.get("scale", 255.0))
    dtype = meta.get("dtype", "uint8")
    if "float" in str(dtype):
        out = arr.astype(np.float32)
    else:
        out = np.rint(arr * scale).astype(dtype)
    if ext in (".tif", ".tiff"):
        import tifffile; tifffile.imwrite(path, out)
    else:
        from PIL import Image; Image.fromarray(out).save(path)

def _downsample(x, scale):
    t = torch.from_numpy(x)[None]
    t = F.interpolate(t, scale_factor=1.0/scale, mode="bicubic",
                      align_corners=False, antialias=True)
    return t[0].numpy()

def degrade(gt, scale, rng, speckle_range=(0.02,0.12), gauss_range=(0.01,0.08)):
    sigma_s = rng.uniform(*speckle_range)
    sigma_g = rng.uniform(*gauss_range)
    ops = ["speckle", "gauss", "down"]
    rng.shuffle(ops)
    x = gt.astype(np.float32)
    for op in ops:
        if op == "speckle":
            x = x + x * rng.normal(0.0, sigma_s, x.shape).astype(np.float32)
        elif op == "gauss":
            x = x + rng.normal(0.0, sigma_g, x.shape).astype(np.float32)
        else:
            x = _downsample(x, scale)
    return x.astype(np.float32)

class PairedDataset(Dataset):
    def __init__(self, pairs, scale=2, crop=128, augment=True,
                 synth_prob=0.5, seed=0,
                 speckle_range=(0.02,0.12), gauss_range=(0.01,0.08)):
        self.pairs  = pairs
        self.scale  = scale; self.crop = crop; self.augment = augment
        self.synth_prob = synth_prob; self.seed = seed
        self.speckle_range = speckle_range; self.gauss_range = gauss_range
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        gt_path, lr_path = self.pairs[idx]
        gt, _ = read_image(gt_path)
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = np.random.default_rng((self.seed, idx, wid, random.randrange(1<<30)))
        if self.synth_prob>0 and rng.random()<self.synth_prob:
            lr = degrade(gt, self.scale, rng, self.speckle_range, self.gauss_range)
        else:
            lr, _ = read_image(lr_path)
        if self.crop is not None:
            c = self.crop
            _, h, w = lr.shape
            if h < c or w < c:
                # pad rather than crash on small images
                ph = max(0, c-h); pw = max(0, c-w)
                lr = np.pad(lr, ((0,0),(0,ph),(0,pw)), mode="reflect")
                gt = np.pad(gt, ((0,0),(0,ph*self.scale),(0,pw*self.scale)), mode="reflect")
                _, h, w = lr.shape
            y = int(rng.integers(0, h-c+1))
            x = int(rng.integers(0, w-c+1))
            lr = lr[:, y:y+c, x:x+c]
            gt = gt[:, y*self.scale:(y+c)*self.scale, x*self.scale:(x+c)*self.scale]
        if self.augment:
            if rng.random()<0.5: lr, gt = lr[:,:,::-1].copy(), gt[:,:,::-1].copy()
            if rng.random()<0.5: lr, gt = lr[:,::-1,:].copy(), gt[:,::-1,:].copy()
            k = int(rng.integers(0,4))
            if k:
                lr = np.rot90(lr, k, (1,2)).copy()
                gt = np.rot90(gt, k, (1,2)).copy()
        return (torch.from_numpy(np.ascontiguousarray(lr)),
                torch.from_numpy(np.ascontiguousarray(gt)))

def split_pairs(pairs, val_frac=0.1, seed=1234):
    pairs = sorted(pairs)
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(pairs)*val_frac)))
    val_idx = set(idx[:n_val])
    train = [pairs[i] for i in idx[n_val:]]
    val   = [pairs[i] for i in sorted(val_idx)]
    return train, val
