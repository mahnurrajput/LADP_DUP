"""
analyze_loss_normalisation.py — LADP (audit M2)
===============================================
g_phi is trained on COCO; LADP is evaluated on VizWiz.  The robust loss
normalisation  L̄ = clip((L - p5) / (p95 - p5), 0, 1)  uses p5/p95 measured on
the model's own COCO *training* patches.  This script measures how per-patch
loss is distributed on

  coco_train   a sample of g_phi's training images            (in-sample)
  coco_val     the validation images (never trained on)      (same domain)
  vizwiz_calib VizWiz val *tuning* images, half A             (target domain, for fitting)
  vizwiz_eval  VizWiz val *tuning* images, half B (disjoint)  (target domain, for judging)

and compares normalisation strategies on coco_val and vizwiz_eval:

  robust_stored        p5/p95 stored in norm_stats.pt (COCO train, in-sample) — current
  robust_coco_train    p5/p95 re-measured on the coco_train sample (sanity check of the above)
  robust_coco_val      p5/p95 from coco_val
  robust_vizwiz_calib  p5/p95 from vizwiz_calib (judged on the disjoint vizwiz_eval)
  per_image            min-max within each image

For each: share of patches clipped to 0 and to 1 (a clipped patch loses its
rank among the other clipped patches — ties), mean/std of L̄, per-image mean
and within-image spread.

No leakage: VizWiz images come only from indices < data.vizwiz_heldout_start;
calibration and evaluation halves are disjoint; nothing is fitted on coco_val
or vizwiz_eval except the strategy named after them (reported, never used to
judge itself).  Existing results are never overwritten (new timestamped folder).

Outputs (results/normalisation_analysis/<run_name>_<time>/):
  analysis.json                    all numbers
  calibration_vizwiz_tuning.json   p5/p95 from vizwiz_calib, usable as
                                   loss.norm_calibration (opt-in; configs are not changed)

  python scripts/analyze_loss_normalisation.py --config configs/pilot_mlp3_small.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from src.config_paths import project_root_for, resolve_config_paths  # noqa: E402

PCTS = (1, 5, 25, 50, 75, 95, 99)
SAT_EPS = 1e-6


# ---------------------------------------------------------------------------
# Pure helpers (tested)
# ---------------------------------------------------------------------------

def percentiles(L: np.ndarray) -> Dict[str, float]:
    v = np.percentile(np.asarray(L, dtype=np.float64).ravel(), PCTS)
    return {f"p{p}": float(x) for p, x in zip(PCTS, v)}


def robust(L: np.ndarray, p5: float, p95: float) -> np.ndarray:
    return np.clip((L - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)


def per_image(L: np.ndarray) -> np.ndarray:
    mn, mx = L.min(axis=1, keepdims=True), L.max(axis=1, keepdims=True)
    return (L - mn) / (mx - mn + 1e-8)


def score_stats(Lbar: np.ndarray) -> Dict[str, float]:
    """Lbar: (n_images, n_patches) normalised scores."""
    lo, hi = Lbar <= SAT_EPS, Lbar >= 1 - SAT_EPS
    img_mean = Lbar.mean(axis=1)
    return {
        "sat0_pct": float(100 * lo.mean()), "sat1_pct": float(100 * hi.mean()),
        "clipped_pct": float(100 * (lo | hi).mean()),
        "images_half_sat1_pct": float(100 * (hi.mean(axis=1) >= 0.5).mean()),
        "images_half_sat0_pct": float(100 * (lo.mean(axis=1) >= 0.5).mean()),
        "mean": float(Lbar.mean()), "std": float(Lbar.std()),
        "image_mean_p5": float(np.percentile(img_mean, 5)), "image_mean_p50": float(np.median(img_mean)),
        "image_mean_p95": float(np.percentile(img_mean, 95)),
        "within_image_std": float(Lbar.std(axis=1).mean()),
    }


def split_vizwiz(n_available: int, heldout_start: int, n: int, seed: int = 0):
    """Disjoint calibration / evaluation index halves from the tuning range [0, heldout_start)."""
    pool = min(n_available, heldout_start)
    if pool < 2:
        raise ValueError("need at least 2 tuning examples")
    n = min(n, pool)
    idx = np.random.default_rng(seed).permutation(pool)[:n]
    half = n // 2
    calib, evals = np.sort(idx[:half]), np.sort(idx[half:])
    assert calib.max(initial=-1) < heldout_start and evals.max(initial=-1) < heldout_start
    return calib.tolist(), evals.tolist()


def compare(sources: Dict[str, np.ndarray], stored: tuple) -> Dict:
    """Every strategy evaluated on coco_val and vizwiz_eval."""
    ranges = {"robust_stored": stored}
    for name in ("coco_train", "coco_val", "vizwiz_calib"):
        if name in sources:
            ranges[f"robust_{name}"] = (float(np.percentile(sources[name], 5)),
                                        float(np.percentile(sources[name], 95)))
    out = {"ranges": {k: {"p5": v[0], "p95": v[1]} for k, v in ranges.items()}, "eval": {}}
    for target in ("coco_val", "vizwiz_eval"):
        if target not in sources:
            continue
        L = sources[target]
        res = {k: score_stats(robust(L, *v)) for k, v in ranges.items()}
        res["per_image"] = score_stats(per_image(L))
        out["eval"][target] = res
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def coco_sets(cache_dir: Path, ckpt_dir: Path, cfg: dict, root: Path, val_split: Optional[str]):
    """(train_files, val_files) used by the run (split.json if present, else the pinned val + first N)."""
    files = sorted(cache_dir.glob("features_*.pt"))
    split = ckpt_dir / "split.json"
    if split.exists():
        s = json.loads(split.read_text(encoding="utf-8"))
        return [cache_dir / f for f in s["train"]], [cache_dir / f for f in s["val"]], str(split)
    vs = val_split or cfg.get("training", {}).get("fixed_val_images") or "configs/splits/coco_pilot_val500.json"
    vs = Path(vs) if Path(vs).is_absolute() else root / vs
    val = set(json.loads(vs.read_text(encoding="utf-8"))["files"])
    mi = cfg.get("data", {}).get("max_images")
    pool = files[: int(mi)] if mi else files
    return [f for f in pool if f.name not in val], [cache_dir / f for f in sorted(val)], str(vs)


@torch.no_grad()
def coco_losses(est, files: List[Path]) -> np.ndarray:
    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        rows.append(est.compute(d["E"].float(), d["T"].float()).cpu().numpy())
    return np.stack(rows)


@torch.no_grad()
def vizwiz_losses(est, vp, image_paths: List[Path]) -> np.ndarray:
    from PIL import Image
    rows = []
    for p in image_paths:
        with Image.open(p) as im:
            px = vp.image_processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"]
        E, T = vp(px)                                        # same extraction as the cache (batch 1)
        rows.append(est.compute(E[0].float(), T[0].float()).cpu().numpy())
    return np.stack(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Loss normalisation under COCO -> VizWiz domain shift (M2).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_coco_train", type=int, default=300)
    ap.add_argument("--n_coco_val", type=int, default=300)
    ap.add_argument("--n_vizwiz", type=int, default=600, help="tuning images, split half/half calib/eval")
    ap.add_argument("--coco_val_split", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf_home", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    import run_diagnostic as rd                                 # same loading + pairing checks
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = project_root_for(args.config)
    rc = resolve_config_paths(cfg, root)
    paths = rd.resolve_run_paths(rd.parse_args(["--config", args.config]), cfg, args.config)
    run = str(cfg.get("training", {}).get("run_name", "recon"))
    out = root / "results" / "normalisation_analysis" / f"{run}_{time.strftime('%Y%m%d_%H%M%S')}"
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")

    device = torch.device(args.device)
    est = rd.load_loss_estimator(paths["recon_checkpoint"], paths["norm_stats"], cfg, device)
    from loss_estimator import state_dict_fingerprint
    sha = state_dict_fingerprint(est.recon_net.state_dict())

    cache_dir = Path(rc["data"]["cache_dir"]) / "coco_train"
    train_f, val_f, split_src = coco_sets(cache_dir, Path(paths["recon_checkpoint"]).parent, cfg, root,
                                          args.coco_val_split)
    rng = np.random.default_rng(args.seed)
    pick = lambda fs, n: [fs[i] for i in sorted(rng.permutation(len(fs))[:n])]  # noqa: E731
    sources = {}
    t0 = time.time()
    sources["coco_train"] = coco_losses(est, pick(train_f, args.n_coco_train))
    sources["coco_val"] = coco_losses(est, pick(val_f, args.n_coco_val))
    print(f"COCO losses: {sources['coco_train'].shape[0]} train / {sources['coco_val'].shape[0]} val images "
          f"[{time.time() - t0:.0f} s]")

    vqa = json.load(open(paths["vizwiz_questions"], encoding="utf-8"))
    heldout = int(cfg.get("data", {}).get("vizwiz_heldout_start", 3800))
    calib_idx, eval_idx = split_vizwiz(len(vqa), heldout, args.n_vizwiz, args.seed)
    from onthefly_features import VisionProjector
    vp = VisionProjector(hf_home=args.hf_home, device=args.device, dtype=torch.float16)
    img = lambda i: Path(paths["vizwiz_images"]) / vqa[i]["image"]  # noqa: E731
    t0 = time.time()
    sources["vizwiz_calib"] = vizwiz_losses(est, vp, [img(i) for i in calib_idx])
    sources["vizwiz_eval"] = vizwiz_losses(est, vp, [img(i) for i in eval_idx])
    print(f"VizWiz losses: {len(calib_idx)} calib / {len(eval_idx)} eval tuning images [{time.time() - t0:.0f} s]")

    ns = est.norm_stats
    result = {
        "config": args.config, "checkpoint": paths["recon_checkpoint"], "model_sha256": sha,
        "coco_split_source": split_src, "seed": args.seed, "vizwiz_heldout_start": heldout,
        "vizwiz_calib_indices": calib_idx, "vizwiz_eval_indices": eval_idx,
        "loss_percentiles": {k: percentiles(v) for k, v in sources.items()},
        "image_mean_loss_percentiles": {k: percentiles(v.mean(axis=1)) for k, v in sources.items()},
        "n_images": {k: int(v.shape[0]) for k, v in sources.items()},
        "comparison": compare(sources, (ns.L_p5, ns.L_p95)),
    }
    out.mkdir(parents=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    cal = result["comparison"]["ranges"]["robust_vizwiz_calib"]
    (out / "calibration_vizwiz_tuning.json").write_text(json.dumps({
        "L_p5": cal["p5"], "L_p95": cal["p95"], "model_sha256": sha,
        "source": f"VizWiz val tuning calibration half ({len(calib_idx)} images, seed {args.seed}, "
                  f"indices < {heldout})", "indices": calib_idx}, indent=1), encoding="utf-8")

    print("\nper-patch loss percentiles (p5 / p50 / p95):")
    for k, v in result["loss_percentiles"].items():
        print(f"  {k:13s} {v['p5']:.4f} / {v['p50']:.4f} / {v['p95']:.4f}")
    for target, res in result["comparison"]["eval"].items():
        print(f"\non {target}:  strategy            clipped0%  clipped1%  mean   within-img std")
        for k, s in res.items():
            print(f"  {k:22s} {s['sat0_pct']:8.1f} {s['sat1_pct']:9.1f}  {s['mean']:.3f}  {s['within_image_std']:.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
