"""
Tests for resumable/idempotent caching (scripts/cache_features.py) and for the
full-COCO split rules in ReconstructionTrainer (fixed pilot validation images,
train-only normalisation, subset_mode, require_complete_cache, warmup_ratio).
No GPU, no real model: a fake extractor produces correctly shaped fp16 tensors.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loss_estimator import ReconstructionNet, ReconstructionTrainer  # noqa: E402

_spec = importlib.util.spec_from_file_location("cache_features", ROOT / "scripts" / "cache_features.py")
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


# ---------------------------------------------------------------------------
# Fake extractor + images
# ---------------------------------------------------------------------------

class _Proc:
    def __call__(self, images, return_tensors="pt"):
        import numpy as np
        a = torch.from_numpy(np.asarray(images.resize((4, 4)), dtype=np.float32)).mean()
        return {"pixel_values": torch.full((1, 3, 8, 8), float(a))}


class FakeExtractor:
    image_processor = _Proc()

    def __init__(self):
        self.calls = 0

    def __call__(self, px):
        self.calls += px.shape[0]
        b = px.shape[0]
        v = px.flatten(1).mean(1).view(b, 1, 1)
        return (v.expand(b, 576, 1024).half().contiguous(),
                (v * 2).expand(b, 576, 4096).half().contiguous())


def _images(d: Path, n: int):
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = d / f"{i:012d}.jpg"
        Image.new("RGB", (16, 16), (i * 20 % 255, 10, 200)).save(p)
        paths.append(p)
    return paths


def _run(paths, out, ext, **kw):
    kw.setdefault("num_workers", 0)
    kw.setdefault("batch_size", 2)
    kw.setdefault("min_free_gb", 0.0)
    return cf.cache_split(ext, paths, out, **kw)


# ---------------------------------------------------------------------------
# cache_features.py
# ---------------------------------------------------------------------------

class TestCacheFeatures:
    def test_builds_all_and_is_idempotent(self, tmp_path):
        paths = _images(tmp_path / "img", 5)
        out = tmp_path / "cache"
        ext = FakeExtractor()
        entries, errors, done = _run(paths, out, ext)
        assert done and not errors and len(entries) == 5 and ext.calls == 5
        assert sorted(p.name for p in out.glob("features_*.pt")) == [f"features_{i:06d}.pt" for i in range(5)]
        assert not list(out.glob("*.tmp"))
        ext2 = FakeExtractor()
        entries, _, done = _run(paths, out, ext2)          # second run: nothing to do
        assert done and len(entries) == 5 and ext2.calls == 0

    def test_resume_reuses_valid_and_replaces_corrupt(self, tmp_path):
        paths = _images(tmp_path / "img", 6)
        out = tmp_path / "cache"
        _run(paths[:4], out, FakeExtractor())              # "existing 10k" = first 4
        bad = out / "features_000002.pt"
        bad.write_bytes(bad.read_bytes()[:1000])           # truncated file
        (out / "features_000005.pt.tmp").write_bytes(b"partial")   # crash leftover
        ext = FakeExtractor()
        entries, errors, done = _run(paths, out, ext, reverify=True)
        assert done and not errors and len(entries) == 6
        assert ext.calls == 3                               # 2 (corrupt), 4, 5
        assert not list(out.glob("*.tmp"))
        assert cf.verify_cache_file(bad, 2, paths[2]) is None

    def test_verify_detects_wrong_image_shape_and_index(self, tmp_path):
        paths = _images(tmp_path / "img", 2)
        out = tmp_path / "cache"
        _run(paths, out, FakeExtractor())
        f0 = out / "features_000000.pt"
        assert cf.verify_cache_file(f0, 0, paths[0]) is None
        assert "image mismatch" in cf.verify_cache_file(f0, 0, paths[1])
        assert "index mismatch" in cf.verify_cache_file(f0, 1, paths[0])
        d = torch.load(f0, weights_only=True)
        d["E"] = d["E"][:10]
        torch.save(d, f0)
        assert "shape" in cf.verify_cache_file(f0, 0, paths[0])
        assert "unreadable" in cf.verify_cache_file(out / "nope.pt", 0, paths[0])

    def test_full_verify_detects_nonfinite(self, tmp_path):
        paths = _images(tmp_path / "img", 1)
        out = tmp_path / "cache"
        _run(paths, out, FakeExtractor())
        f0 = out / "features_000000.pt"
        d = torch.load(f0, weights_only=True)
        d["T"][0, 0] = float("nan")
        torch.save(d, f0)
        assert cf.verify_cache_file(f0, 0, paths[0], "quick") is None
        assert "non-finite" in cf.verify_cache_file(f0, 0, paths[0], "full")

    def test_trusted_index_skips_reverification_until_file_changes(self, tmp_path, monkeypatch):
        paths = _images(tmp_path / "img", 3)
        out = tmp_path / "cache"
        _run(paths, out, FakeExtractor())
        _run(paths, out, FakeExtractor())                   # records "quick" for all
        calls = []
        real = cf.verify_cache_file
        monkeypatch.setattr(cf, "verify_cache_file", lambda *a, **k: calls.append(a[0].name) or real(*a, **k))
        _run(paths, out, FakeExtractor())
        assert calls == []
        f1 = out / "features_000001.pt"
        os.utime(f1, ns=(f1.stat().st_atime_ns, f1.stat().st_mtime_ns + 10**9))
        _run(paths, out, FakeExtractor())
        assert calls == ["features_000001.pt"]

    def test_fails_early_when_not_enough_space(self, tmp_path, monkeypatch):
        paths = _images(tmp_path / "img", 4)
        out = tmp_path / "cache"
        monkeypatch.setattr(cf, "free_gb", lambda p: 10.0)  # 4 files need ~0.024 GB
        ext = FakeExtractor()
        with pytest.raises(SystemExit) as e:
            _run(paths, out, ext, min_free_gb=40.0)
        assert e.value.code == 2 and ext.calls == 0

    def test_stops_cleanly_when_space_drops(self, tmp_path, monkeypatch):
        paths = _images(tmp_path / "img", 8)
        out = tmp_path / "cache"
        free = iter([100.0] + [100.0] + [1.0] * 100)        # start ok, then below margin
        monkeypatch.setattr(cf, "free_gb", lambda p: next(free))
        entries, _, done = _run(paths, out, FakeExtractor(), min_free_gb=40.0,
                                space_check_every=2, batch_size=2)
        assert not done and 0 < len(entries) < 8
        _run(paths, out, FakeExtractor())                   # resume completes
        assert len(list(out.glob("features_*.pt"))) == 8

    def test_bad_image_is_reported_not_fatal(self, tmp_path):
        paths = _images(tmp_path / "img", 3)
        paths[1].write_bytes(b"not a jpeg")
        entries, errors, done = _run(paths, tmp_path / "cache", FakeExtractor())
        assert done and len(entries) == 2 and [e["index"] for e in errors] == [1]


# ---------------------------------------------------------------------------
# Trainer split rules
# ---------------------------------------------------------------------------

DV, DT = 32, 64


def _cache(d: Path, n: int, manifest_complete: bool | None = None):
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    files = []
    for i in range(n):
        f = d / f"features_{i:06d}.pt"
        torch.save({"E": torch.randn(4, DV, generator=g), "T": torch.randn(4, DT, generator=g)}, f)
        files.append({"filename": f.name, "image_path": f"img/{i:012d}.jpg"})
    if manifest_complete is not None:
        (d / "manifest.json").write_text(json.dumps({
            "total_images": n, "cached_ok": n if manifest_complete else n - 1,
            "complete": manifest_complete, "files": files}))
    return d


def _fixed(tmp_path, names, images=None):
    p = tmp_path / "fixed.json"
    p.write_text(json.dumps({"files": names,
                             "images": images or [f"{int(n[9:15]):012d}.jpg" for n in names]}))
    return str(p)


def _trainer(tmp_path, data=None, **training):
    t = {"epochs": 1, "batch_size": 8, "val_fraction": 0.1, "files_per_shard": 4, "seed": 42}
    t.update(training)
    cfg = {"checkpoints_dir": str(tmp_path / "ck"), "training": t, "data": data or {}}
    torch.manual_seed(0)
    return ReconstructionTrainer(ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=16, depth="mlp2"), cfg)


def _split(tmp_path):
    s = json.loads((tmp_path / "ck" / "recon" / "split.json").read_text())
    return s["train"], s["val"], s["val_ref"]


class TestSplit:
    def test_fixed_val_images_stay_in_val_and_val_is_topped_up(self, tmp_path):
        cache = _cache(tmp_path / "c", 40, manifest_complete=True)
        fixed = [f"features_{i:06d}.pt" for i in (1, 7, 30)]
        t = _trainer(tmp_path, fixed_val_images=_fixed(tmp_path, fixed))
        t.fit(str(cache))
        train, val, ref = _split(tmp_path)
        assert set(fixed) <= set(val) and sorted(ref) == sorted(fixed)
        assert len(val) == 4 and len(train) == 36            # int(40 * 0.1) = 4
        assert not set(train) & set(val)
        assert len(t.val_ref_loss_history) == 1

    def test_pilot_split_preserved_when_cache_grows(self, tmp_path):
        """Split on the first 20 files, then on 60 files with the 20-file val pinned."""
        small = _cache(tmp_path / "small", 20)
        _trainer(tmp_path, val_fraction=0.1).fit(str(small))
        _, pilot_val, _ = _split(tmp_path)
        big = _cache(tmp_path / "big", 60)
        _trainer(tmp_path, fixed_val_images=_fixed(tmp_path, pilot_val)).fit(str(big))
        train, val, _ = _split(tmp_path)
        assert set(pilot_val) <= set(val) and not set(pilot_val) & set(train)
        # and the first-N mode on the big cache reproduces the pilot split exactly
        _trainer(tmp_path, data={"max_images": 20, "subset_mode": "first"},
                 fixed_val_images=_fixed(tmp_path, pilot_val)).fit(str(big))
        train2, val2, _ = _split(tmp_path)
        assert sorted(val2) == sorted(pilot_val) and len(train2) == 18

    def test_normalisation_stats_use_train_split_only(self, tmp_path, monkeypatch):
        cache = _cache(tmp_path / "c", 20)
        t = _trainer(tmp_path)
        seen = {}
        real = t._streaming_mean_std

        def spy(loader):
            seen["files"] = [os.path.basename(p) for p in loader.file_paths]
            return real(loader)

        monkeypatch.setattr(t, "_streaming_mean_std", spy)
        t.fit(str(cache))
        train, val, _ = _split(tmp_path)
        assert sorted(seen["files"]) == sorted(train) and not set(seen["files"]) & set(val)

    def test_fixed_val_mismatch_with_manifest_raises(self, tmp_path):
        cache = _cache(tmp_path / "c", 10, manifest_complete=True)
        bad = _fixed(tmp_path, ["features_000001.pt"], images=["999999999999.jpg"])
        with pytest.raises(ValueError, match="different images"):
            _trainer(tmp_path, fixed_val_images=bad).fit(str(cache))

    def test_require_complete_cache(self, tmp_path):
        # Since the audit fix F1, "use all images" (max_images: null) with
        # require_complete_cache also needs data.expected_images.
        req = {"require_complete_cache": True, "expected_images": 10}
        cache = _cache(tmp_path / "c", 10, manifest_complete=False)
        with pytest.raises(RuntimeError, match="incomplete"):
            _trainer(tmp_path, data=req).fit(str(cache))
        cache2 = _cache(tmp_path / "c2", 10, manifest_complete=True)
        _trainer(tmp_path, data=req).fit(str(cache2))
        with pytest.raises(ValueError, match="expected_images"):
            _trainer(tmp_path, data={"require_complete_cache": True}).fit(str(cache2))

    def test_warmup_ratio_sets_warmup_steps(self, tmp_path):
        cache = _cache(tmp_path / "c", 20)          # 18 train files x 4 patches / batch 8 = 9 steps/epoch
        t = _trainer(tmp_path, epochs=4, scheduler="warmup_cosine", warmup_ratio=0.25, warmup_steps=2000)
        t.fit(str(cache))
        assert t.warmup_steps == 9                  # round(0.25 * 36)
