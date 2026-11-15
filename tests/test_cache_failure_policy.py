"""
Audit M5: explicit failure policy and completion criteria for the feature cache
(scripts/cache_features.py) and how the trainer treats exclusions.
No GPU: a fake extractor produces correctly shaped fp16 tensors.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from loss_estimator import ReconstructionNet, ReconstructionTrainer  # noqa: E402

_spec = importlib.util.spec_from_file_location("cache_features_m5", ROOT / "scripts" / "cache_features.py")
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


class _Proc:
    def __call__(self, images, return_tensors="pt"):
        a = float(np.asarray(images.resize((4, 4)), dtype=np.float32).mean())
        return {"pixel_values": torch.full((1, 3, 8, 8), a)}


class FakeExtractor:
    image_processor = _Proc()

    def __init__(self):
        self.seen = 0

    def __call__(self, px):
        self.seen += px.shape[0]
        v = px.flatten(1).mean(1).view(-1, 1, 1)
        return v.expand(-1, 576, 1024).half().contiguous(), v.expand(-1, 576, 4096).half().contiguous()


def _images(d: Path, n: int, broken=()):
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = d / f"{i:012d}.jpg"
        if i in broken:
            p.write_bytes(b"not a jpeg")
        else:
            Image.new("RGB", (16, 16), (i * 30 % 255, 10, 200)).save(p)
        paths.append(p)
    return paths


def _run(paths, out, prev=({}, {}), exclude=False, min_attempts=2, retry_excluded=False, **kw):
    """One cache_features run: extract, then write the manifest like main() does."""
    pf, pe = prev
    ext = FakeExtractor()
    kw.setdefault("min_free_gb", 0.0)
    entries, errors, completed = cf.cache_split(
        ext, paths, out, batch_size=2, num_workers=0,
        skip_indices=set() if retry_excluded else set(pe), **kw)
    m = cf.build_manifest(paths, entries, errors, pf, pe, stopped_early=not completed,
                          exclude_failed=exclude, min_attempts=min_attempts, retry_excluded=retry_excluded)
    cf.atomic_json_dump(m, out / "manifest.json")
    return m, ext, cf.load_previous_manifest(out / "manifest.json")


class TestFailurePolicy:
    def test_unreadable_image_is_recorded_not_skipped_silently(self, tmp_path):
        paths = _images(tmp_path / "img", 4, broken={2})
        m, _, _ = _run(paths, tmp_path / "c")
        assert m["counts"] == {"total": 4, "cached": 3, "failed": 1, "excluded": 0, "pending": 0,
                               "retried_ok_this_run": 0}
        f = m["failed"][0]
        assert f["index"] == 2 and f["attempts"] == 1 and "Error" in f["error"]
        assert m["complete"] is False and m["errors"] == 1
        assert "completion_criteria" in m

    def test_failed_images_are_retried_and_attempts_accumulate(self, tmp_path):
        paths = _images(tmp_path / "img", 4, broken={2})
        _, _, prev = _run(paths, tmp_path / "c")
        m, ext, _ = _run(paths, tmp_path / "c", prev=prev)
        assert m["failed"][0]["attempts"] == 2 and m["complete"] is False
        assert ext.seen == 0                      # the 3 good files were reused, only #2 retried (and failed)

    def test_exclusion_needs_flag_and_min_attempts(self, tmp_path):
        paths = _images(tmp_path / "img", 4, broken={2})
        m1, _, prev = _run(paths, tmp_path / "c", exclude=True, min_attempts=2)
        assert m1["counts"]["excluded"] == 0 and m1["complete"] is False      # only 1 attempt so far
        m2, _, _ = _run(paths, tmp_path / "c", prev=prev, exclude=True, min_attempts=2)
        assert m2["counts"]["excluded"] == 1 and m2["counts"]["failed"] == 0
        e = m2["excluded"][0]
        assert e["index"] == 2 and e["attempts"] == 2 and "failed 2 attempt(s)" in e["reason"]
        assert m2["complete"] is True and m2["cached_ok"] == 3

    def test_excluded_image_not_retried_unless_asked(self, tmp_path):
        paths = _images(tmp_path / "img", 3, broken={1})
        _, _, prev = _run(paths, tmp_path / "c", min_attempts=1, exclude=True)
        assert prev[1] and not prev[0]                                         # excluded after 1 attempt
        Image.new("RGB", (16, 16)).save(paths[1])                              # image repaired
        m, ext, prev2 = _run(paths, tmp_path / "c", prev=prev)
        assert ext.seen == 0 and m["counts"]["excluded"] == 1 and m["complete"] is True
        m3, ext3, _ = _run(paths, tmp_path / "c", prev=prev2, retry_excluded=True)
        assert ext3.seen == 1 and m3["counts"] == {"total": 3, "cached": 3, "failed": 0, "excluded": 0,
                                                    "pending": 0, "retried_ok_this_run": 0}

    def test_repaired_image_is_recovered(self, tmp_path):
        paths = _images(tmp_path / "img", 3, broken={0})
        _, _, prev = _run(paths, tmp_path / "c")
        Image.new("RGB", (16, 16)).save(paths[0])
        m, ext, _ = _run(paths, tmp_path / "c", prev=prev)
        assert ext.seen == 1 and m["counts"]["retried_ok_this_run"] == 1
        assert m["counts"]["failed"] == 0 and m["complete"] is True

    def test_stopped_early_leaves_pending_and_incomplete(self, tmp_path, monkeypatch):
        paths = _images(tmp_path / "img", 6)
        free = iter([1000.0, 1000.0, 0.0] + [0.0] * 20)                       # space runs out mid-run
        monkeypatch.setattr(cf, "free_gb", lambda p: next(free))
        m, _, _ = _run(paths, tmp_path / "c", min_free_gb=1.0, space_check_every=2)
        assert m["stopped_early"] is True and m["complete"] is False
        assert m["counts"]["pending"] > 0 and m["counts"]["cached"] + m["counts"]["pending"] == 6

    def test_loop_finishing_is_not_enough(self, tmp_path):
        paths = _images(tmp_path / "img", 2)
        m = cf.build_manifest(paths, [{"index": 0}], [], {}, {}, stopped_early=False)
        assert m["complete"] is False and m["counts"]["pending"] == 1

    def test_min_attempts_validated(self, tmp_path):
        with pytest.raises(ValueError):
            cf.build_manifest([], [], [], {}, {}, False, min_attempts=0)

    def test_unreadable_previous_manifest_is_tolerated(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text("{broken")
        assert cf.load_previous_manifest(p) == ({}, {})
        assert cf.load_previous_manifest(tmp_path / "none.json") == ({}, {})


# ---------------------------------------------------------------------------
# Trainer: exclusions must be explicit
# ---------------------------------------------------------------------------

def _small_cache(d: Path, n: int, manifest: dict):
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        if i not in {e["index"] for e in manifest.get("excluded", [])}:
            torch.save({"E": torch.randn(4, 8, generator=g), "T": torch.randn(4, 12, generator=g)},
                       d / f"features_{i:06d}.pt")
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def _trainer(ck, expected):
    cfg = {"checkpoints_dir": str(ck), "data": {"require_complete_cache": True, "expected_images": expected},
           "training": {"epochs": 1, "batch_size": 8, "val_fraction": 0.2, "files_per_shard": 2, "seed": 0}}
    return ReconstructionTrainer(ReconstructionNet(d_in=12, d_out=8, hidden_dim=6, depth="mlp2"), cfg)


def test_trainer_accepts_complete_cache_with_explicit_exclusions(tmp_path):
    paths = [Path(f"{i:012d}.jpg") for i in range(10)]
    entries = [{"index": i, "filename": f"features_{i:06d}.pt", "image_path": str(p)}
               for i, p in enumerate(paths) if i != 3]
    m = cf.build_manifest(paths, entries, [], {3: {"index": 3, "image_path": "x", "error": "bad", "attempts": 2}},
                          {}, stopped_early=False, exclude_failed=True, min_attempts=2)
    assert m["complete"] and m["counts"]["excluded"] == 1
    cache = _small_cache(tmp_path / "c", 10, m)
    t = _trainer(tmp_path / "ck", expected=10)
    t.fit(str(cache))
    assert len(t.val_loss_history) == 1


def test_trainer_refuses_failed_images_that_are_not_excluded(tmp_path):
    paths = [Path(f"{i:012d}.jpg") for i in range(10)]
    entries = [{"index": i, "filename": f"features_{i:06d}.pt", "image_path": str(p)}
               for i, p in enumerate(paths) if i != 3]
    m = cf.build_manifest(paths, entries, [{"index": 3, "image_path": "x", "error": "bad"}], {}, {},
                          stopped_early=False)
    assert m["complete"] is False
    cache = _small_cache(tmp_path / "c", 10, m)
    (cache / "features_000003.pt").unlink()          # the failed image has no file
    with pytest.raises(RuntimeError, match="incomplete"):
        _trainer(tmp_path / "ck", expected=10).fit(str(cache))


def test_trainer_refuses_tampered_manifest(tmp_path):
    # complete=true but cached + excluded does not add up to total
    m = {"total_images": 10, "cached_ok": 8, "complete": True, "files": [], "excluded": [{"index": 3}]}
    cache = _small_cache(tmp_path / "c", 10, m)
    with pytest.raises(RuntimeError, match="incomplete"):
        _trainer(tmp_path / "ck", expected=10).fit(str(cache))
