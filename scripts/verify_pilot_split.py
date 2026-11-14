"""
verify_pilot_split.py — LADP
============================
Checks that configs/splits/coco_pilot_val500.json is really the pilot's
validation set: recomputes the validation MSE of the pilot's best.pt on those
500 images, with the pilot's own normalisation statistics and the trainer's
exact formula, and compares it with the best val_loss in the pilot log.
Read-only.

  python scripts/verify_pilot_split.py
"""

import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from loss_estimator import NormStats, ReconstructionNet, ShardedPatchLoader  # noqa: E402


def main() -> None:
    ck = ROOT / "checkpoints" / "recon_pilot"
    if not (ck / "best.pt").exists():
        ck = ROOT / "checkpoints" / "recon"
    log = ROOT / "results" / "pilot_mlp3_small" / "train_recon.log"
    if not log.exists():
        log = ROOT / "results" / "stage1_diagnostic" / "train_recon.log"
    split = json.loads((ROOT / "configs" / "splits" / "coco_pilot_val500.json").read_text())
    files = [str(ROOT / "data" / "cache" / "coco_train" / f) for f in split["files"]]

    vals = [float(m.group(1)) for m in re.finditer(r"val_loss=([\d.]+)", log.read_text(encoding="utf-8", errors="ignore"))]
    logged_best = min(vals)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ReconstructionNet(depth="mlp3_small")
    net.load_state_dict(torch.load(ck / "best.pt", map_location="cpu", weights_only=True))
    net.to(dev).eval()
    ns = NormStats.load(str(ck / "norm_stats.pt"))
    Em, Es, Tm, Ts = (x.to(dev) for x in (ns.E_mean, ns.E_std, ns.T_mean, ns.T_std))

    tot, n = 0.0, 0
    with torch.no_grad():
        for e, t in ShardedPatchLoader(files, 128, 64, shuffle=False, seed=42, device=dev):
            e, t = e.float(), t.float()
            E = (e - Em) / (Es + 1e-8)
            T = (t - Tm) / (Ts + 1e-8)
            tot += F.mse_loss(net(T), E, reduction="sum").item()
            n += E.numel()
    recomputed = tot / n
    ok = abs(recomputed - logged_best) <= 1e-5
    print(json.dumps({"checkpoint": str(ck / "best.pt"), "log": str(log), "n_val_images": len(files),
                      "logged_best_val_loss": logged_best, "recomputed_val_loss": recomputed,
                      "abs_diff": abs(recomputed - logged_best), "match": ok}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
