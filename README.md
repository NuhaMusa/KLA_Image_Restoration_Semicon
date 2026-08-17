# KLA Hackathon — AI-Based Restoration of Degraded Images

## Overview
NAFNet-style U-Net restoration model for joint denoising (speckle + Gaussian)
and 2x super-resolution on grayscale semiconductor inspection images.

- Architecture: NAFNet U-Net (encoder-bottleneck-decoder, 4.95M params)
- Loss: Charbonnier + SSIM (w_char=1.0, w_ssim=0.2)
- Best validation PSNR: 29.150 dB (+5.87 dB over bicubic baseline)
- Best validation SSIM: 0.8028

## Environment Setup
```bash
pip install -r requirements.txt
```

## Running Inference
```bash
python inference.py --input_dir <path_to_degraded_images> --output_dir <path_to_save_restored>
```
Optional arguments:
- `--weights` (default: `weights/best.pth`)
- `--precision` (`fp32`/`fp16`/`bf16`, default: `fp16`)
- `--batch_size` (default: 16)
- `--device` (default: `cuda`)

## Input/Output Contract
- Input: grayscale `.npy` files, values roughly in [0, 1.5] (speckle noise may
  push values slightly outside [0,1] -- this is expected and handled internally)
- Output: grayscale `.npy` files, values clamped to [0, 1], at 2x the input resolution
- Filenames are preserved between input and output directories

## Training
See `training.ipynb` (or the Kaggle notebook this repo was exported from) for
the full training pipeline. Key details:
- Dataset: KLA-provided paired GT/NoisyLR images (3200 pairs, 90/10 train/val split)
- On-the-fly synthetic degradation augmentation (speckle + Gaussian + downsample,
  randomized order and severity) alongside the fixed on-disk degraded pairs
- 150 epochs, AdamW, cosine LR schedule, EMA (decay 0.999), fp16 mixed precision
- Trained on Kaggle T4 GPU

## Assumptions
- Input images are single-channel grayscale
- Fixed 2x scale factor between degraded and ground-truth resolution
- No manual source-code edits required to run `inference.py`

## Results

Evaluated on a held-out validation split (320 images, 10% of the training
data, never used for training or model selection).

| Metric | Bicubic Baseline | Our Model | Improvement |
|---|---|---|---|
| PSNR  | 23.279 dB | 29.150 dB | +5.871 dB |
| SSIM  | 0.5629    | 0.8028    | +0.2399 |
| LPIPS | 0.4312    | 0.2390    | -0.1922 (lower is better) |

**Inference throughput** (Kaggle T4 GPU, batch_size=16, fp16): 10.5 ms/image,
measured on the official 400-image test set (`kla-restoration-NoisyLR`).

See `results/sample_comparison.png` for a validation-set example and
`results/test_set_spot_check.png` for real test-set restorations (including
one strong case and one honest failure case with visible remaining blur on
high-frequency textured content).
