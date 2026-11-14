"""
export_pilot_val_split.py — LADP
================================
Re-derives the FYP-1 pilot's validation split and writes it to
configs/splits/coco_pilot_val500.json, so later runs (e.g. full COCO) can keep
exactly the same 500 validation images (training.fixed_val_images).

The pilot (14 Nov 2026) split the 10,000 cached files with the trainer's rule:
    files = sorted(features_*.pt)                       # 10,000 files
    perm  = torch.randperm(10000, Generator().manual_seed(42))
    val   = files[perm[:int(10000 * 0.05)]]             # 500 images
Only file names and the manifest are read; no tensors.

  python scripts/export_pilot_val_split.py
"""

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "coco_train"
OUT = ROOT / "configs" / "splits" / "coco_pilot_val500.json"
N_PILOT, SEED, VAL_FRAC = 10_000, 42, 0.05


def main() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    fmap = {r["filename"]: os.path.basename(r["image_path"]) for r in manifest["files"]}
    files = [f"features_{i:06d}.pt" for i in range(N_PILOT)]       # == sorted glob of the pilot cache
    missing = [f for f in files if f not in fmap]
    if missing:
        sys.exit(f"manifest lacks {len(missing)} pilot files, e.g. {missing[:3]}")
    perm = torch.randperm(N_PILOT, generator=torch.Generator().manual_seed(SEED)).tolist()
    n_val = min(N_PILOT - 1, max(1, int(N_PILOT * VAL_FRAC)))
    val = sorted(files[i] for i in perm[:n_val])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "description": "FYP-1 pilot g_phi validation images (500 of the first 10,000 sorted "
                       "COCO train2017 images; seed 42, val_fraction 0.05). Keep these in "
                       "validation for all later runs so results stay comparable.",
        "n_pilot_images": N_PILOT, "seed": SEED, "val_fraction": VAL_FRAC,
        "files": val, "images": [fmap[f] for f in val],
    }, indent=1), encoding="utf-8")
    print(f"wrote {len(val)} validation images to {OUT}")


if __name__ == "__main__":
    main()
