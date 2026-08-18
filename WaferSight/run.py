#!/usr/bin/env python3
"""
KLA Image Restoration — Entry Script
Usage: python run.py <input-dir> <output-dir>
"""
import sys, os, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import read_image, save_image
from src.model import build_model
import torch

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python run.py <input-dir> <output-dir>")

    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    weights = str(Path(__file__).parent / "models" / "best.pth")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt    = torch.load(weights, map_location="cpu", weights_only=False)
    model   = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    # Warmup
    if device.type == "cuda":
        with torch.inference_mode():
            model(torch.zeros(1, 1, 128, 128, device=device))
        torch.cuda.synchronize()

    # Find input files
    in_files = sorted(Path(input_dir).glob("*.npy"))
    if not in_files:
        sys.exit(f"No .npy files found in {input_dir}")
    print(f"Restoring {len(in_files)} images...")

    # Run inference
    t0 = time.perf_counter()
    amp_dtype = torch.float16 if device.type == "cuda" else None

    with torch.inference_mode():
        for path in in_files:
            arr, meta = read_image(str(path))
            t   = torch.from_numpy(arr)[None].to(device)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(t)
            # Save as (H, W) float32 in [0, 1]
            out_np = out.float().clamp(0, 1).squeeze().cpu().numpy()
            out_path = os.path.join(output_dir, path.name)
            np.save(out_path, out_np.astype(np.float32))

    elapsed = time.perf_counter() - t0
    print(f"Done: {len(in_files)} images in {elapsed:.2f}s "
          f"({1000*elapsed/len(in_files):.1f} ms/img)")
    print(f"device={device}  torch={torch.__version__}")

    # Verify outputs
    print("Verifying outputs...")
    for path in in_files:
        out_f = Path(output_dir) / path.name
        assert out_f.exists(), f"Missing output: {path.name}"
        arr = np.load(str(out_f))
        assert arr.ndim in (2, 3), f"{path.name}: bad shape {arr.shape}"
        assert arr.min() >= 0.0 and arr.max() <= 1.0,             f"{path.name}: values out of [0,1]"
        assert np.isfinite(arr).all(), f"{path.name}: contains NaN/Inf"
    print(f"Verified {len(in_files)} outputs — all pass.")

if __name__ == "__main__":
    main()
