"""Audit M2: helpers of scripts/analyze_loss_normalisation.py (no GPU, synthetic losses)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("aln", ROOT / "scripts" / "analyze_loss_normalisation.py")
aln = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aln)
import loss_estimator as le  # noqa: E402


def test_split_vizwiz_is_disjoint_and_tuning_only():
    c, e = aln.split_vizwiz(4319, 3800, 600, seed=0)
    assert len(c) == 300 and len(e) == 300 and not set(c) & set(e)
    assert max(c + e) < 3800
    c2, e2 = aln.split_vizwiz(4319, 3800, 10_000)           # capped at the tuning pool
    assert len(c2) + len(e2) == 3800 and max(c2 + e2) < 3800
    assert aln.split_vizwiz(4319, 3800, 600, seed=0) == (c, e)      # reproducible
    with pytest.raises(ValueError):
        aln.split_vizwiz(1, 3800, 10)


def test_robust_and_per_image():
    L = np.array([[0.0, 0.1, 0.3, 0.5, 2.0]])
    assert np.allclose(aln.robust(L, 0.1, 0.5), [[0, 0, 0.5, 1, 1]], atol=1e-6)
    pi = aln.per_image(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert np.allclose(pi[0], [0, 0.5, 1]) and np.allclose(pi[1], 0)


def test_score_stats_saturation():
    Lbar = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5]])
    s = aln.score_stats(Lbar)
    assert s["sat0_pct"] == 25.0 and s["sat1_pct"] == 62.5 and s["clipped_pct"] == 87.5
    assert s["images_half_sat1_pct"] == 100.0 and s["images_half_sat0_pct"] == 50.0


def test_compare_detects_domain_shift():
    rng = np.random.default_rng(0)
    coco = rng.gamma(2.0, 0.03, (50, 576))                   # low losses
    viz = rng.gamma(2.0, 0.09, (50, 576))                    # shifted target domain
    src = {"coco_train": coco, "coco_val": coco * 1.05, "vizwiz_calib": viz, "vizwiz_eval": viz[::-1]}
    stored = (float(np.percentile(coco, 5)), float(np.percentile(coco, 95)))
    r = aln.compare(src, stored)
    v = r["eval"]["vizwiz_eval"]
    assert v["robust_stored"]["sat1_pct"] > 30                 # COCO range saturates on VizWiz
    assert abs(v["robust_vizwiz_calib"]["sat1_pct"] - 5) < 2  # target-domain range ~5% by construction
    assert v["per_image"]["sat1_pct"] < 1
    assert set(r["ranges"]) == {"robust_stored", "robust_coco_train", "robust_coco_val", "robust_vizwiz_calib"}


def test_calibration_file_is_usable_by_the_estimator(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"L_p5": 0.05, "L_p95": 0.4, "model_sha256": "abc", "source": "test"}))
    ns = le.NormStats(torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2), 0.02, 0.14,
                      {"model_sha256": "abc"})
    out = le.apply_norm_calibration(ns, str(cal))
    assert (out.L_p5, out.L_p95) == (0.05, 0.4)


def test_coco_sets_from_pinned_split(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(10):
        (cache / f"features_{i:06d}.pt").write_bytes(b"")
    vs = tmp_path / "val.json"
    vs.write_text(json.dumps({"files": ["features_000002.pt", "features_000007.pt"]}))
    tr, va, src = aln.coco_sets(cache, tmp_path / "ck", {"data": {"max_images": 8}}, tmp_path, str(vs))
    assert [f.name for f in va] == ["features_000002.pt", "features_000007.pt"]
    assert [f.name for f in tr] == [f"features_{i:06d}.pt" for i in (0, 1, 3, 4, 5, 6)]  # first 8 minus val 2, 7
    assert not set(f.name for f in tr) & set(f.name for f in va)
