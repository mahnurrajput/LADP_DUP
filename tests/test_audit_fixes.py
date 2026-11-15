"""
Tests for the fixes made after the implementation audit (docs/AUDIT_mlp3_paper.md):

  F1  legacy / incomplete / too-small caches cannot pass as the full COCO cache
  F2  no silent fallback to mlp3_small; configs build the intended model size
  F3  steps/epoch counted per shard, so the lr schedule ends exactly at the end
  F4  weights-only (or scheduler-less) resume continues the schedule, no warm-up replay
  F5  normalisation statistics are saved and reused on resume
  F6  statistics saved before training; finalise-only resume; atomic NormStats
  F7  resume refuses a checkpoint from a different split / batch / model / optimiser
  F8  mlp3_paper with a non-2048 hidden width is flagged
  F9  missing training keys are reported
  F10 failed sanity check -> False (train_recon exits non-zero)
  F11 missing best.pt is caught before training; non-finite loss stops the run

Read-only checks against the real repository (configs, 10k cache manifest, pilot
checkpoint) are skipped if those files are absent.  Nothing in the repository is
written: all runs use pytest's tmp_path.
"""

import importlib.util
import json
import logging
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import loss_estimator as le  # noqa: E402
from loss_estimator import (  # noqa: E402
    NormStats, ReconstructionNet, ReconstructionTrainer, ShardedPatchLoader,
    build_recon_net_from_config, COCO_TRAIN2017_IMAGES, MLP3_PAPER_LLAVA_PARAMS,
    MLP3_SMALL_LLAVA_PARAMS,
)

DV, DT = 32, 64
PAPER = dict(optimizer="adamw", scheduler="warmup_cosine", warmup_steps=3, min_lr_ratio=0.01,
             grad_clip=1.0, loss_reduction="mean", patience=10, min_delta=1e-3)


def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cache(d: Path, n_files=8, n=8, manifest=None, seed=0):
    """Synthetic cache. manifest: None | 'legacy' | 'complete' | 'incomplete'."""
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    files = []
    for i in range(n_files):
        f = d / f"features_{i:06d}.pt"
        torch.save({"E": torch.randn(n, DV, generator=g), "T": torch.randn(n, DT, generator=g)}, f)
        files.append({"filename": f.name, "image_path": f"img/{i:012d}.jpg"})
    if manifest:
        m = {"total_images": n_files, "cached_ok": n_files, "files": files}
        if manifest == "complete":
            m["complete"] = True
        elif manifest == "incomplete":
            m.update(complete=False, cached_ok=n_files - 1)
        (d / "manifest.json").write_text(json.dumps(m))
    return d


def _cfg(ck, data=None, run_name="recon", **training):
    t = {"epochs": 4, "batch_size": 16, "patience": 10, "val_fraction": 0.25, "lr": 1e-3,
         "files_per_shard": 2, "seed": 42, **PAPER, "run_name": run_name}
    t.update(training)
    return {"checkpoints_dir": str(ck), "training": t, "data": data or {}}


def _net(depth="mlp3_paper"):
    torch.manual_seed(123)
    return ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=48, depth=depth)


def _trainer(ck, **kw):
    depth = kw.pop("depth", "mlp3_paper")
    return ReconstructionTrainer(_net(depth), _cfg(ck, **kw))


def _crash_at(trainer, epoch_no):
    """Make trainer raise KeyboardInterrupt at the start of its epoch_no-th epoch."""
    orig, calls = trainer._train_epoch, []

    def crashing(loader, opt):
        calls.append(1)
        if len(calls) == epoch_no:
            raise KeyboardInterrupt
        return orig(loader, opt)

    trainer._train_epoch = crashing


def _last(ck, run="recon"):
    return torch.load(Path(ck) / run / "last.pt", weights_only=True)


# ---------------------------------------------------------------------------
# F1  cache completeness
# ---------------------------------------------------------------------------

class TestF1CacheCompleteness:
    def test_legacy_manifest_without_complete_flag_is_rejected(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, manifest="legacy")   # total == cached_ok, no 'complete'
        t = _trainer(tmp_path / "ck", data={"require_complete_cache": True, "expected_images": 8})
        with pytest.raises(RuntimeError, match="no 'complete' flag"):
            t.fit(str(cache))

    def test_incomplete_manifest_is_rejected(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, manifest="incomplete")
        t = _trainer(tmp_path / "ck", data={"require_complete_cache": True, "expected_images": 8})
        with pytest.raises(RuntimeError, match="incomplete"):
            t.fit(str(cache))

    def test_complete_but_smaller_cache_is_rejected(self, tmp_path):
        # a complete 8-image cache (like the complete 10k cache) for a run that needs 12
        cache = _cache(tmp_path / "c", 8, manifest="complete")
        t = _trainer(tmp_path / "ck", data={"require_complete_cache": True, "expected_images": 12})
        with pytest.raises(RuntimeError, match="needs 12"):
            t.fit(str(cache))

    def test_all_images_without_expected_count_is_rejected(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, manifest="complete")
        t = _trainer(tmp_path / "ck", data={"require_complete_cache": True})
        with pytest.raises(ValueError, match="expected_images"):
            t.fit(str(cache))

    def test_expected_count_checked_even_without_require_flag(self, tmp_path):
        cache = _cache(tmp_path / "c", 8)
        t = _trainer(tmp_path / "ck", data={"expected_images": 12})
        with pytest.raises(RuntimeError, match="expected_images=12"):
            t.fit(str(cache))

    def test_max_images_larger_than_cache_is_rejected_with_require(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, manifest="complete")
        t = _trainer(tmp_path / "ck", data={"require_complete_cache": True, "max_images": 12})
        with pytest.raises(RuntimeError, match="needs 12"):
            t.fit(str(cache))

    def test_full_complete_cache_is_accepted(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, manifest="complete")
        t = _trainer(tmp_path / "ck", epochs=1,
                     data={"require_complete_cache": True, "expected_images": 8})
        t.fit(str(cache))
        assert len(t.val_loss_history) == 1

    def test_refusal_happens_before_anything_is_written(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, manifest="legacy")
        t = _trainer(tmp_path / "ck", data={"require_complete_cache": True, "expected_images": 8})
        with pytest.raises(RuntimeError):
            t.fit(str(cache))
        assert not (tmp_path / "ck").exists()

    def test_real_10k_cache_cannot_pass_as_full_coco(self, tmp_path):
        """The real base config + the real 10k pilot cache must be refused (read-only)."""
        cache = ROOT / "data" / "cache" / "coco_train"
        cfg_path = ROOT / "configs" / "base_llava_vizwiz.yaml"
        if not (cache / "manifest.json").exists() or not cfg_path.exists():
            pytest.skip("real cache or config not present")
        n_files = len(list(cache.glob("features_*.pt")))
        if n_files >= COCO_TRAIN2017_IMAGES:
            pytest.skip("full cache already built")
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["data"]["max_images"] is None
        assert cfg["data"]["require_complete_cache"] is True
        assert cfg["data"]["expected_images"] == COCO_TRAIN2017_IMAGES
        cfg["checkpoints_dir"] = str(tmp_path / "ck")           # never the real checkpoints
        cfg["training"]["fixed_val_images"] = None
        t = ReconstructionTrainer(ReconstructionNet(depth="mlp3_paper"), cfg, device="cpu")
        with pytest.raises(RuntimeError):
            t.fit(str(cache))
        assert not (tmp_path / "ck").exists()
        # and the manifest check alone, with the required count
        with pytest.raises(RuntimeError):
            ReconstructionTrainer._check_cache_complete(str(cache), n_files, COCO_TRAIN2017_IMAGES)


# ---------------------------------------------------------------------------
# F2 / F8  model construction from config
# ---------------------------------------------------------------------------

class TestF2ModelFromConfig:
    def test_missing_reconstructor_raises(self):
        with pytest.raises(ValueError, match="loss.reconstructor is missing"):
            build_recon_net_from_config({"loss": {"hidden_dim": 2048}})

    def test_misspelt_key_is_named(self):
        with pytest.raises(ValueError, match="reconstrutor"):
            build_recon_net_from_config({"loss": {"reconstrutor": "mlp3_paper"}})

    def test_no_loss_section_raises(self):
        with pytest.raises(ValueError):
            build_recon_net_from_config({})

    def test_unknown_depth_raises(self):
        with pytest.raises(ValueError):
            build_recon_net_from_config({"loss": {"reconstructor": "mlp3_papr"}})

    def test_expected_params_mismatch_raises(self):
        with pytest.raises(ValueError, match="expected_params"):
            build_recon_net_from_config({"loss": {"reconstructor": "mlp3_small",
                                                  "expected_params": MLP3_PAPER_LLAVA_PARAMS}})

    def test_paper_model_exact_size(self):
        net = build_recon_net_from_config({"loss": {"reconstructor": "mlp3_paper", "hidden_dim": 2048,
                                                    "expected_params": MLP3_PAPER_LLAVA_PARAMS}})
        assert net.depth == "mlp3_paper" and net.param_count() == 27_282_432

    @pytest.mark.parametrize("cfg_name,depth,params", [
        ("base_llava_vizwiz.yaml", "mlp3_paper", 27_282_432),
        ("paper_10k_mlp3_paper.yaml", "mlp3_paper", 27_282_432),
        ("pilot_mlp3_small.yaml", "mlp3_small", 14_685_184),
    ])
    def test_real_configs_build_the_intended_model(self, cfg_name, depth, params):
        p = ROOT / "configs" / cfg_name
        if not p.exists():
            pytest.skip(f"{cfg_name} not present")
        cfg = yaml.safe_load(p.read_text())
        assert cfg["loss"]["expected_params"] == params
        net = build_recon_net_from_config(cfg)
        assert net.depth == depth and net.param_count() == params
        assert sum(p.numel() for p in net.parameters()) == params

    def test_mlp3_paper_layer_structure_unchanged(self):
        n = ReconstructionNet(depth="mlp3_paper").net
        shapes = [(m.in_features, m.out_features) for m in n if isinstance(m, torch.nn.Linear)]
        assert shapes == [(4096, 4096), (4096, 2048), (2048, 1024)]
        assert [type(m).__name__ for m in n] == ["Linear", "LayerNorm", "GELU", "Dropout",
                                                 "Linear", "LayerNorm", "GELU", "Dropout", "Linear"]
        assert MLP3_PAPER_LLAVA_PARAMS == 27_282_432 and MLP3_SMALL_LLAVA_PARAMS == 14_685_184

    def test_f8_wrong_hidden_dim_warns_and_expected_params_catches_it(self, caplog):
        with caplog.at_level(logging.WARNING, logger="loss_estimator"):
            net = ReconstructionNet(depth="mlp3_paper", hidden_dim=1024)
        assert "not the 27,282,432" in caplog.text and net.param_count() == 22_036_480
        with pytest.raises(ValueError, match="expected_params"):
            build_recon_net_from_config({"loss": {"reconstructor": "mlp3_paper", "hidden_dim": 1024,
                                                  "expected_params": MLP3_PAPER_LLAVA_PARAMS}})

    def test_train_recon_cli_refuses_missing_reconstructor(self, tmp_path):
        import shutil
        import subprocess
        (tmp_path / "src").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "configs").mkdir()
        shutil.copy(ROOT / "src" / "loss_estimator.py", tmp_path / "src")
        shutil.copy(ROOT / "scripts" / "train_recon.py", tmp_path / "scripts")
        _cache(tmp_path / "data" / "cache" / "coco_train", 4)
        cfg = {"output_dir": "results/run", "data": {"cache_dir": "data/cache"},
               "model": {"device": "cpu"}, "loss": {"hidden_dim": 16}, "training": {"epochs": 1}}
        (tmp_path / "configs" / "c.yaml").write_text(yaml.safe_dump(cfg))
        p = subprocess.run([sys.executable, "scripts/train_recon.py", "--config", "configs/c.yaml"],
                           cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        assert p.returncode == 2 and "loss.reconstructor is missing" in p.stdout + p.stderr
        assert not (tmp_path / "checkpoints").exists()


# ---------------------------------------------------------------------------
# F3  steps per epoch
# ---------------------------------------------------------------------------

class TestF3StepCount:
    @pytest.mark.parametrize("n_files,fps,ppf,bs", [
        (10, 3, 8, 16), (10, 2, 8, 32), (7, 7, 576, 73728 // 64), (12, 4, 576, 4 * 576),
        (5, 64, 4, 3), (9, 1, 5, 5)])
    def test_num_batches_equals_actual_iteration(self, tmp_path, n_files, fps, ppf, bs):
        paths = []
        for i in range(n_files):
            p = tmp_path / f"features_{i:06d}.pt"
            torch.save({"E": torch.zeros(ppf, 2), "T": torch.zeros(ppf, 3)}, p)
            paths.append(str(p))
        loader = ShardedPatchLoader(paths, bs, fps, shuffle=True, seed=0)
        assert loader.num_batches(ppf) == sum(1 for _ in loader)

    def test_full_coco_layout(self):
        loader = ShardedPatchLoader(["x"] * 112_373, 73_728, 128)
        assert loader.num_batches(576) == 878                   # 877 full + 1 partial
        bad = ShardedPatchLoader(["x"] * 112_373, 73_728, 64)
        assert bad.num_batches(576) == 1756                     # was assumed 878 before the fix

    def test_schedule_ends_at_floor_on_last_step_with_misaligned_shards(self, tmp_path, caplog):
        # 6 train files x 8 patches, 2 files/shard -> 16-patch shards, batch 32:
        # one 16-patch batch per shard = 3 steps/epoch (old count: ceil(48/32) = 2)
        cache = _cache(tmp_path / "c", 8, 8)
        t = _trainer(tmp_path / "ck", epochs=3, batch_size=32, files_per_shard=2, warmup_steps=2)
        lrs = []
        orig = t._train_epoch

        def spy(loader, opt):
            out = orig(loader, opt)
            lrs.append(opt.param_groups[0]["lr"])
            return out

        t._train_epoch = spy
        with caplog.at_level(logging.WARNING, logger="loss_estimator"):
            t.fit(str(cache))
        info = _last(tmp_path / "ck")["run_info"]
        assert info["steps_per_epoch"] == 3 and info["total_steps"] == 9
        assert t._last_epoch_steps == 3
        assert "not a multiple of batch_size" in caplog.text
        assert lrs[-1] == pytest.approx(0.01 * 1e-3)            # floor reached exactly at the end
        assert lrs[-2] > 0.02 * 1e-3                            # ...and not before
        assert "optimiser steps but the lr schedule assumes" not in caplog.text


# ---------------------------------------------------------------------------
# F4  scheduler on resume
# ---------------------------------------------------------------------------

class TestF4SchedulerResume:
    def _ref_and_crash(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, 8)
        ref = _trainer(tmp_path / "a", epochs=4)
        ref.fit(str(cache))
        crash = _trainer(tmp_path / "b", epochs=4)
        _crash_at(crash, 3)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        return cache, _last(tmp_path / "a")

    def test_weights_only_resume_does_not_replay_warmup(self, tmp_path):
        cache, ref_last = self._ref_and_crash(tmp_path)
        t = _trainer(tmp_path / "b", epochs=4)
        lr_at_start = []
        orig = t._train_epoch

        def spy(loader, opt):
            lr_at_start.append(opt.param_groups[0]["lr"])
            return orig(loader, opt)

        t._train_epoch = spy
        t.fit(str(cache), resume_from=str(tmp_path / "b" / "recon" / "best.pt"),
              start_epoch=3, best_val_loss=10.0)
        res_last = _last(tmp_path / "b")
        spe = res_last["run_info"]["steps_per_epoch"]
        assert res_last["scheduler"]["last_epoch"] == ref_last["scheduler"]["last_epoch"] == 4 * spe
        assert res_last["optimizer"]["param_groups"][0]["lr"] == pytest.approx(
            ref_last["optimizer"]["param_groups"][0]["lr"])
        # lr at the start of epoch 3 = schedule value at step 2*spe, not warm-up step 0
        assert lr_at_start[0] == pytest.approx(1e-3 * t._scheduler.lr_lambdas[0](2 * spe))
        assert lr_at_start[0] > 0

    def test_full_checkpoint_without_scheduler_state_is_fast_forwarded(self, tmp_path):
        cache, ref_last = self._ref_and_crash(tmp_path)
        p = tmp_path / "b" / "recon" / "last.pt"
        ck = torch.load(p, weights_only=True)
        ck["scheduler"] = None                                   # pilot-era checkpoint
        torch.save(ck, p)
        _trainer(tmp_path / "b", epochs=4).fit(str(cache), resume_from=str(p))
        res_last = _last(tmp_path / "b")
        assert res_last["scheduler"]["last_epoch"] == ref_last["scheduler"]["last_epoch"]
        assert res_last["optimizer"]["param_groups"][0]["lr"] == pytest.approx(
            ref_last["optimizer"]["param_groups"][0]["lr"])

    def test_make_scheduler_last_step_equals_stepping(self, tmp_path):
        t = _trainer(tmp_path / "ck")
        o1 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        s1 = t._make_scheduler(o1, 50)
        for _ in range(17):
            o1.step()
            s1.step()
        o2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        t._make_scheduler(o2, 50)                                # sets initial_lr like fit() does
        s2 = t._make_scheduler(o2, 50, last_step=17)
        assert s2.last_epoch == s1.last_epoch == 17
        assert o2.param_groups[0]["lr"] == pytest.approx(o1.param_groups[0]["lr"])

    def test_exact_resume_from_last_pt_still_bit_identical(self, tmp_path):
        cache, ref_last = self._ref_and_crash(tmp_path)
        _trainer(tmp_path / "b", epochs=4).fit(
            str(cache), resume_from=str(tmp_path / "b" / "recon" / "last.pt"))
        res_last = _last(tmp_path / "b")
        for k in ref_last["model"]:
            assert torch.equal(ref_last["model"][k], res_last["model"][k]), k
        assert res_last["history"] == ref_last["history"]


# ---------------------------------------------------------------------------
# F5 / F6  normalisation statistics
# ---------------------------------------------------------------------------

def _count_stats_passes(t):
    calls = []
    orig = t._streaming_mean_std

    def spy(loader):
        calls.append(1)
        return orig(loader)

    t._streaming_mean_std = spy
    return calls


class TestF5F6Stats:
    def test_resume_reuses_saved_stats(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, 8)
        crash = _trainer(tmp_path / "ck", epochs=4)
        _crash_at(crash, 3)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        t = _trainer(tmp_path / "ck", epochs=4)
        calls = _count_stats_passes(t)
        t.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
        assert calls == []
        assert torch.equal(t._E_mean, crash._E_mean) and torch.equal(t._T_std_v, crash._T_std_v)

    def test_stats_saved_before_first_epoch_and_used_by_weights_only_resume(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, 8)
        crash = _trainer(tmp_path / "ck", epochs=4)
        _crash_at(crash, 1)                                      # dies during epoch 1
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        s = torch.load(tmp_path / "ck" / "recon" / "train_stats.pt", weights_only=True)
        assert torch.equal(s["E_mean"], crash._E_mean) and s["n_train"] == 6 * 8
        # a weights-only resume (no stats in that file) still finds train_stats.pt
        w = tmp_path / "w.pt"
        torch.save(crash.recon_net.state_dict(), w)
        t = _trainer(tmp_path / "ck", epochs=2)
        calls = _count_stats_passes(t)
        t.fit(str(cache), resume_from=str(w), start_epoch=1, best_val_loss=float("inf"))
        assert calls == []

    def test_changed_train_file_forces_recompute(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, 8)
        _trainer(tmp_path / "ck", epochs=1).fit(str(cache))
        _trainer(tmp_path / "ck", epochs=1).fit(str(cache))    # same files -> same split
        split = json.loads((tmp_path / "ck" / "recon" / "split.json").read_text())
        f = cache / split["train"][0]
        d = torch.load(f, weights_only=True)
        d["E"] = d["E"] + 1.0
        torch.save(d, f)
        os.utime(f, ns=(os.stat(f).st_atime_ns, os.stat(f).st_mtime_ns + 10_000_000))
        t = _trainer(tmp_path / "ck", epochs=2, allow_resume_mismatch=True)
        calls = _count_stats_passes(t)
        t.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
        assert calls == [1]

    def test_finalise_only_resume_after_crash_in_p5_p95(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, 8)
        t = _trainer(tmp_path / "ck", epochs=2)

        def boom(loader):
            raise KeyboardInterrupt

        t._compute_train_losses = boom
        with pytest.raises(KeyboardInterrupt):
            t.fit(str(cache))
        ns_path = tmp_path / "ck" / "recon" / "norm_stats.pt"
        assert not ns_path.exists() and (tmp_path / "ck" / "recon" / "best.pt").exists()
        t2 = _trainer(tmp_path / "ck", epochs=2)
        calls = _count_stats_passes(t2)
        trained = []
        orig = t2._train_epoch
        t2._train_epoch = lambda l, o: (trained.append(1), orig(l, o))[1]
        _, ns = t2.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
        assert trained == [] and calls == [] and ns_path.exists()
        assert ns.L_p5 <= ns.L_p95 and len(t2.val_loss_history) == 2

    def test_early_stopped_checkpoint_resume_runs_no_more_epochs(self, tmp_path, monkeypatch):
        cache = _cache(tmp_path / "c", 8, 8)
        t = _trainer(tmp_path / "ck", epochs=6, patience=2)
        vals = iter([1.0, 2.0, 2.0])
        monkeypatch.setattr(t, "_val_epoch", lambda loader: next(vals))
        t.fit(str(cache))
        assert len(t.val_loss_history) == 3                     # stopped by patience
        t2 = _trainer(tmp_path / "ck", epochs=6, patience=2)
        trained = []
        orig = t2._train_epoch
        t2._train_epoch = lambda l, o: (trained.append(1), orig(l, o))[1]
        t2.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
        assert trained == []

    def test_norm_stats_save_is_atomic(self, tmp_path):
        ns = NormStats(torch.zeros(3), torch.ones(3), torch.zeros(4), torch.ones(4), 0.1, 0.9)
        p = tmp_path / "ns.pt"
        ns.save(str(p))
        assert p.exists() and not (tmp_path / "ns.pt.tmp").exists()
        assert NormStats.load(str(p)).L_p95 == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# F7  resume compatibility
# ---------------------------------------------------------------------------

class TestF7ResumeChecks:
    def _first(self, tmp_path, **kw):
        cache = _cache(tmp_path / "c", 8, 8)
        _trainer(tmp_path / "ck", epochs=2, **kw).fit(str(cache))
        return cache, str(tmp_path / "ck" / "recon" / "last.pt")

    @pytest.mark.parametrize("change,key", [
        ({"batch_size": 8}, "batch_size"),
        ({"val_fraction": 0.5}, "split_fp"),
        ({"lr": 5e-4}, "lr"),
        ({"optimizer": "adam"}, "optimizer"),
        ({"loss_reduction": "sum"}, "loss_reduction"),
        ({"files_per_shard": 3}, "files_per_shard"),
    ])
    def test_mismatch_is_refused(self, tmp_path, change, key):
        cache, last = self._first(tmp_path)
        t = _trainer(tmp_path / "ck", epochs=4, **change)
        with pytest.raises(ValueError, match=key):
            t.fit(str(cache), resume_from=last)

    def test_different_model_is_refused_before_loading(self, tmp_path):
        cache, last = self._first(tmp_path)
        t = _trainer(tmp_path / "ck", epochs=4, depth="mlp3_small")
        with pytest.raises(ValueError, match="depth"):
            t.fit(str(cache), resume_from=last)

    def test_override_flag_allows_mismatch(self, tmp_path):
        cache, last = self._first(tmp_path)
        t = _trainer(tmp_path / "ck", epochs=3, batch_size=8, allow_resume_mismatch=True)
        t.fit(str(cache), resume_from=last)
        assert len(t.val_loss_history) == 3

    def test_more_epochs_only_warns(self, tmp_path, caplog):
        cache, last = self._first(tmp_path)
        with caplog.at_level(logging.WARNING, logger="loss_estimator"):
            t = _trainer(tmp_path / "ck", epochs=3)
            t.fit(str(cache), resume_from=last)
        assert "epochs changed 2 -> 3" in caplog.text and len(t.val_loss_history) == 3

    def test_old_checkpoint_without_run_info_warns_and_works(self, tmp_path, caplog):
        cache, last = self._first(tmp_path)
        ck = torch.load(last, weights_only=True)
        del ck["run_info"]
        torch.save(ck, last)
        with caplog.at_level(logging.WARNING, logger="loss_estimator"):
            _trainer(tmp_path / "ck", epochs=3).fit(str(cache), resume_from=last)
        assert "no run_info" in caplog.text


# ---------------------------------------------------------------------------
# F9  defaults are reported
# ---------------------------------------------------------------------------

def test_f9_missing_training_keys_are_reported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="loss_estimator"):
        ReconstructionTrainer(_net(), {"checkpoints_dir": str(tmp_path), "training": {"lr": 1e-4}})
    assert "training.optimizer" in caplog.text and "training.loss_reduction" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="loss_estimator"):
        ReconstructionTrainer(_net(), _cfg(tmp_path))            # every key given
    assert "not set in the config" not in caplog.text


def test_f9_real_target_configs_set_every_training_key(caplog):
    for name in ("base_llava_vizwiz.yaml", "paper_10k_mlp3_paper.yaml", "pilot_mlp3_small.yaml"):
        p = ROOT / "configs" / name
        if not p.exists():
            continue
        cfg = yaml.safe_load(p.read_text())
        cfg["checkpoints_dir"] = "unused"
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="loss_estimator"):
            ReconstructionTrainer(ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=48, depth="mlp2"),
                                  cfg)
        assert "not set in the config" not in caplog.text, name


# ---------------------------------------------------------------------------
# F10  sanity check
# ---------------------------------------------------------------------------

class TestF10SanityCheck:
    def _mod(self):
        return _load_module("train_recon_under_test", "scripts/train_recon.py")

    def test_returns_false_on_failure(self, tmp_path):
        tr = self._mod()
        ok = tr.sanity_check(tmp_path / "missing.pt", tmp_path / "missing_ns.pt",
                             {"loss": {"reconstructor": "mlp3_paper"}}, torch.device("cpu"))
        assert ok is False

    def test_returns_false_on_wrong_architecture(self, tmp_path):
        tr = self._mod()
        ck = tmp_path / "best.pt"
        torch.save(ReconstructionNet(depth="mlp3_small").state_dict(), ck)
        ns = tmp_path / "ns.pt"
        NormStats(torch.zeros(1024), torch.ones(1024), torch.zeros(4096), torch.ones(4096),
                  0.1, 0.2).save(str(ns))
        assert tr.sanity_check(ck, ns, {"loss": {"reconstructor": "mlp3_paper"}},
                               torch.device("cpu")) is False
        assert tr.sanity_check(ck, ns, {"loss": {"reconstructor": "mlp3_small"}},
                               torch.device("cpu")) is True


# ---------------------------------------------------------------------------
# F11  best.pt and non-finite losses
# ---------------------------------------------------------------------------

class TestF11:
    def test_full_resume_without_best_pt_fails_before_training(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, 8)
        _trainer(tmp_path / "ck", epochs=2).fit(str(cache))
        last = str(tmp_path / "ck" / "recon" / "last.pt")
        t = _trainer(tmp_path / "ck", epochs=4, run_name="other")   # writes to ck/other/
        trained = []
        orig = t._train_epoch
        t._train_epoch = lambda l, o: (trained.append(1), orig(l, o))[1]
        with pytest.raises(FileNotFoundError, match="best.pt"):
            t.fit(str(cache), resume_from=last)
        assert trained == []

    def test_weights_only_resume_from_elsewhere_creates_best_pt(self, tmp_path):
        cache = _cache(tmp_path / "c", 8, 8)
        w = tmp_path / "weights.pt"
        torch.save(_net().state_dict(), w)
        t = _trainer(tmp_path / "ck", epochs=3)
        t.fit(str(cache), resume_from=str(w), start_epoch=3, best_val_loss=1e9)
        assert (tmp_path / "ck" / "recon" / "best.pt").exists()

    def test_non_finite_loss_stops_and_keeps_last_good_epoch(self, tmp_path, monkeypatch):
        cache = _cache(tmp_path / "c", 8, 8)
        t = _trainer(tmp_path / "ck", epochs=4)
        vals = iter([1.0, float("nan")])
        monkeypatch.setattr(t, "_val_epoch", lambda loader: next(vals))
        with pytest.raises(RuntimeError, match="Non-finite"):
            t.fit(str(cache))
        assert _last(tmp_path / "ck")["epoch"] == 1


# ---------------------------------------------------------------------------
# Pilot model still usable (read-only)
# ---------------------------------------------------------------------------

class TestPilotStillUsable:
    CK = ROOT / "checkpoints" / "recon_pilot"

    def test_pilot_checkpoint_loads_with_pilot_config(self):
        if not (self.CK / "best.pt").exists():
            pytest.skip("pilot checkpoint not present")
        cfg = yaml.safe_load((ROOT / "configs" / "pilot_mlp3_small.yaml").read_text())
        net = build_recon_net_from_config(cfg)
        net.load_state_dict(torch.load(self.CK / "best.pt", map_location="cpu", weights_only=True))
        ns = NormStats.load(str(self.CK / "norm_stats.pt"))
        assert ns.L_p5 == pytest.approx(0.020985, abs=1e-6)
        assert ns.L_p95 == pytest.approx(0.142976, abs=1e-6)
        est = le.LossEstimator(net, ns)
        L = est.compute_normalised(torch.randn(6, 1024), torch.randn(6, 4096))
        assert L.shape == (6,) and torch.isfinite(L).all()

    def test_run_diagnostic_loader_accepts_pilot(self):
        if not (self.CK / "best.pt").exists():
            pytest.skip("pilot checkpoint not present")
        rd = _load_module("run_diagnostic_under_test", "scripts/run_diagnostic.py")
        cfg = yaml.safe_load((ROOT / "configs" / "pilot_mlp3_small.yaml").read_text())
        est = rd.load_loss_estimator(str(self.CK / "best.pt"), str(self.CK / "norm_stats.pt"),
                                     cfg, torch.device("cpu"))
        assert est.recon_net.depth == "mlp3_small" and est.recon_net.param_count() == 14_685_184

    def test_pilot_split_unchanged_by_config_pin(self, tmp_path):
        """max_images 10000 + subset_mode first on a 10k cache == the old max_images null."""
        cache = _cache(tmp_path / "c", 20, 4)
        a = ReconstructionTrainer(_net("mlp2"), _cfg(tmp_path / "a", epochs=1, val_fraction=0.05))
        a.fit(str(cache))
        b = ReconstructionTrainer(_net("mlp2"), _cfg(tmp_path / "b", epochs=1, val_fraction=0.05,
                                                     data={"max_images": 20, "subset_mode": "first"}))
        b.fit(str(cache))
        sa = json.loads((tmp_path / "a" / "recon" / "split.json").read_text())
        sb = json.loads((tmp_path / "b" / "recon" / "split.json").read_text())
        assert sa == sb
