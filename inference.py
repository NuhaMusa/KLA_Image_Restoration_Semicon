
#!/usr/bin/env python3
"""
Standalone inference script for KLA hackathon submission.
Usage:
    python inference.py --input_dir <degraded> --output_dir <restored>
No source-code edits required.
"""
import argparse, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from src.dataset import list_images, read_image, save_image
from src.model   import build_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--weights",    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights", "best.pth"))
    p.add_argument("--precision",  default="fp16", choices=["fp32","fp16","bf16"])
    p.add_argument("--device",     default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--compile",    action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    if device.type != "cuda": amp_dtype = None

    ckpt  = torch.load(args.weights, map_location="cpu")
    cfg   = ckpt.get("cfg", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    for q in model.parameters(): q.requires_grad_(False)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() failed, continuing without it: {e}")

    if device.type == "cuda":
        with torch.inference_mode():
            model(torch.zeros(1, cfg.get("in_ch", 1), 128, 128, device=device))
        torch.cuda.synchronize()

    files = list_images(args.input_dir)
    if not files: raise SystemExit(f"No images in {args.input_dir}")
    print(f"Restoring {len(files)} images (batch_size={args.batch_size})...")

    print("Reading images...")
    with ThreadPoolExecutor(max_workers=8) as read_pool:
        loaded = list(read_pool.map(lambda p: (p, *read_image(p)), files))

    groups = {}
    for path, arr, meta in loaded:
        groups.setdefault(arr.shape, []).append((path, arr, meta))

    save_pool = ThreadPoolExecutor(max_workers=4)
    save_futures = []

    t0, count = time.perf_counter(), 0
    with torch.inference_mode():
        for shape, items in groups.items():
            for i in range(0, len(items), args.batch_size):
                batch = items[i:i + args.batch_size]
                arrs = np.stack([b[1] for b in batch])
                batch_tensor = torch.from_numpy(arrs).pin_memory().to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(batch_tensor)
                out_np = out.float().clamp(0, 1).cpu().numpy()

                for j, (path, _, meta) in enumerate(batch):
                    out_path = os.path.join(args.output_dir, os.path.basename(path))
                    save_futures.append(save_pool.submit(save_image, out_path, out_np[j], meta))
                count += len(batch)

    for f in save_futures:
        f.result()
    save_pool.shutdown()

    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"Done: {count} images in {elapsed:.2f}s ({1000*elapsed/count:.1f} ms/img)")
    print(f"device={device}  precision={args.precision}  batch_size={args.batch_size}  torch={torch.__version__}")

if __name__ == "__main__":
    main()
