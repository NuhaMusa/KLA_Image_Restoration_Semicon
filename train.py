"""
Training script for KLA Image Restoration — SEMICON India Hackathon 2026.
Reproduces the submitted checkpoint from scratch.

Usage:
    python train.py --gt_dir data/GT --lr_dir data/NoisyLR

Dataset structure expected:
    data/
        GT/          <- clean ground-truth .npy images
        NoisyLR/     <- degraded low-resolution .npy images

The script automatically creates a 90/10 train/val split.
Best checkpoint is saved to weights/best.pth.
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent))
from src.model   import build_model
from src.losses  import CombinedLoss, psnr, ssim_metric
from src.dataset import list_images, split_pairs, PairedDataset


# ── Default config (matches submitted checkpoint exp01) ───────────────
DEFAULT_CFG = {
    # model
    "in_ch":       1,
    "width":       32,
    "enc_blocks":  [2, 2, 4],
    "mid_blocks":  8,
    "dec_blocks":  [2, 2, 2],
    "scale":       2,
    # data
    "lr_crop":     128,
    "val_frac":    0.10,
    "synth_prob":  0.5,
    "speckle_range": [0.02, 0.15],
    "gauss_range":   [0.01, 0.10],
    # loss
    "w_char":      1.0,
    "w_ssim":      0.2,
    "w_fft":       0.0,
    "w_lpips":     0.0,
    # training
    "epochs":      150,
    "batch_size":  16,
    "lr":          2e-3,
    "lr_min":      1e-6,
    "weight_decay": 0.0,
    "grad_clip":   1.0,
    "amp":         "fp16",
    "ema_decay":   0.999,
    "num_workers": 2,
    "val_every":   5,
    "seed":        42,
}


def ema_update(ema, model, decay=0.999):
    for s, p in zip(ema.state_dict().values(), model.state_dict().values()):
        if s.dtype.is_floating_point:
            s.mul_(decay).add_(p.detach(), alpha=1 - decay)
        else:
            s.copy_(p)


@torch.no_grad()
def validate(model, val_loader, device, amp_dtype):
    model.eval()
    ps, ss, n = 0.0, 0.0, 0
    for lr_t, gt_t in val_loader:
        lr_t, gt_t = lr_t.to(device), gt_t.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(lr_t)
        out = out.float()
        if out.shape[-2:] != gt_t.shape[-2:]:
            out = F.interpolate(out, size=gt_t.shape[-2:],
                                mode="bicubic", align_corners=False)
        ps += psnr(out, gt_t.float())
        ss += ssim_metric(out, gt_t.float())
        n  += 1
    model.train()
    return ps / max(n, 1), ss / max(n, 1)


def train(cfg, gt_dir, lr_dir, save_dir, weights_dir):
    os.makedirs(save_dir,   exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Save config
    (Path(save_dir) / "config.json").write_text(json.dumps(cfg, indent=2))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"fp16": torch.float16,
                 "bf16": torch.bfloat16,
                 "off":  None}[cfg["amp"]]
    if device.type != "cuda":
        amp_dtype = None
    print(f"Training on {device}  AMP={cfg['amp']}")

    # ── Dataset ──────────────────────────────────────────────────────
    gt_files = list_images(gt_dir)
    lr_map   = {os.path.splitext(os.path.basename(p))[0]: p
                for p in list_images(lr_dir)}
    pairs = [(g, lr_map[os.path.splitext(os.path.basename(g))[0]])
             for g in gt_files
             if os.path.splitext(os.path.basename(g))[0] in lr_map]

    if not pairs:
        raise RuntimeError(
            f"No matched GT/LR pairs found.\n"
            f"  GT  dir: {gt_dir}\n"
            f"  LR  dir: {lr_dir}\n"
            "Check that filenames match between the two folders.")

    train_pairs, val_pairs = split_pairs(pairs, cfg["val_frac"], seed=cfg["seed"])
    print(f"Pairs — total: {len(pairs)}  train: {len(train_pairs)}  val: {len(val_pairs)}")

    train_ds = PairedDataset(
        train_pairs, scale=cfg["scale"], crop=cfg["lr_crop"], augment=True,
        synth_prob=cfg["synth_prob"], seed=cfg["seed"],
        speckle_range=tuple(cfg["speckle_range"]),
        gauss_range=tuple(cfg["gauss_range"]))

    val_ds = PairedDataset(
        val_pairs, scale=cfg["scale"], crop=None, augment=False,
        synth_prob=0.0, seed=cfg["seed"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
        drop_last=True, persistent_workers=cfg["num_workers"] > 0)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Train batches/epoch: {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────
    criterion = CombinedLoss(
        cfg["w_char"], cfg["w_ssim"], cfg["w_fft"], cfg["w_lpips"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], betas=(0.9, 0.9))

    total_steps = cfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg["lr_min"])

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["amp"] == "fp16"))

    # ── Log file ──────────────────────────────────────────────────────
    log_path  = Path(save_dir) / "log.csv"
    best_path = Path(weights_dir) / "best.pth"
    last_path = Path(save_dir) / "last.pth"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_psnr", "val_ssim",
             "ema_psnr", "ema_ssim", "secs"])

    best_score = -1e9
    print(f"\nStarting training for {cfg['epochs']} epochs...")
    print("Epoch  | Loss    | Val PSNR | Val SSIM | EMA PSNR | EMA SSIM | LR")
    print("-" * 75)

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0

        for lr_img, gt_img in train_loader:
            lr_img = lr_img.to(device, non_blocking=True).to(
                memory_format=torch.channels_last)
            gt_img = gt_img.to(device, non_blocking=True).to(
                memory_format=torch.channels_last)

            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(lr_img)
                loss, _ = criterion(out.float(), gt_img.float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_update(ema_model, model, cfg["ema_decay"])

            running += float(loss.detach()) * lr_img.size(0)
            seen    += lr_img.size(0)

        train_loss = running / max(seen, 1)
        cur_lr     = scheduler.get_last_lr()[0]
        secs       = time.time() - t0

        vp = vs = ep = es = float("nan")
        tag = ""
        do_val = ((epoch + 1) % cfg["val_every"] == 0
                  or epoch == cfg["epochs"] - 1)

        if do_val:
            vp, vs = validate(model,     val_loader, device, amp_dtype)
            ep, es = validate(ema_model, val_loader, device, amp_dtype)
            score  = max(vp, ep if ep == ep else -1e9)

            if score > best_score:
                best_score = score
                use_ema = (ep == ep and ep >= vp)
                torch.save({
                    "model":    (ema_model if use_ema else model).state_dict(),
                    "cfg":      cfg,
                    "epoch":    epoch,
                    "psnr":     score,
                    "ema_used": bool(use_ema),
                }, best_path)
                tag = " ← best (EMA)" if use_ema else " ← best"

            print(f"{epoch+1:6d} | {train_loss:.5f} | {vp:8.3f} | {vs:8.4f} | "
                  f"{ep:8.3f} | {es:8.4f} | {cur_lr:.2e}{tag}")
        else:
            print(f"{epoch+1:6d} | {train_loss:.5f} | {'':8} | {'':8} | "
                  f"{'':8} | {'':8} | {cur_lr:.2e}")

        # Save last checkpoint for resume
        torch.save({
            "model": model.state_dict(),
            "ema":   ema_model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "epoch": epoch,
            "best":  best_score,
            "cfg":   cfg,
        }, last_path)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch+1, cur_lr, train_loss, vp, vs, ep, es, secs])

    print(f"\nTraining complete. Best val PSNR: {best_score:.3f} dB")
    print(f"Weights saved to: {best_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KLA Image Restoration — Training Script")
    parser.add_argument("--gt_dir",      required=True,
                        help="Path to GT (clean) images folder")
    parser.add_argument("--lr_dir",      required=True,
                        help="Path to NoisyLR (degraded) images folder")
    parser.add_argument("--save_dir",    default="runs/exp01",
                        help="Directory to save logs and config")
    parser.add_argument("--weights_dir", default="weights",
                        help="Directory to save model weights")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--width",       type=int,   default=None)
    parser.add_argument("--amp",         default=None,
                        choices=["fp16", "bf16", "off"])
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    # Override defaults with any CLI args provided
    for key in ["epochs", "batch_size", "lr", "width", "amp"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    train(cfg,
          gt_dir=args.gt_dir,
          lr_dir=args.lr_dir,
          save_dir=args.save_dir,
          weights_dir=args.weights_dir)


if __name__ == "__main__":
    main()
