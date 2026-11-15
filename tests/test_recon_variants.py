"""
Tests for the g_phi variants (mlp3_small / mlp3_paper) and the paper-style
training options (AdamW, warm-up + cosine, grad clipping, min_delta early
stopping, data.max_images) plus resume support.

Reference for mlp3_paper and the schedule: Lost in Embeddings code,
github.com/lyan62/vlm-info-loss, scripts/reconstruct_embeddings_parallel.py
(EmbeddingTransformer with num_layers=3; SequentialLR(LambdaLR, CosineAnnealingLR)).
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loss_estimator import ReconstructionNet, ReconstructionTrainer  # noqa: E402

PILOT_PARAMS = 14_685_184   # 4096->2048->2048->1024
PAPER_PARAMS = 27_282_432   # 4096->4096->2048->1024 + 2 LayerNorms


# ---------------------------------------------------------------------------
# Reference copies of the source code (verbatim structure)
# ---------------------------------------------------------------------------

def _source_mlp3(input_dim=4096, output_dim=1024, hidden_dim=2048):
    hidden_dim = min(hidden_dim, max(input_dim, output_dim))
    return nn.Sequential(
        nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
    )


def _source_scheduler(opt, lr, warmup, total):
    main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup, eta_min=lr * 0.01)
    warm = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(1.0, s / warmup))
    return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, main], milestones=[warmup])


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class TestArchitectures:
    def test_mlp3_small_param_count(self):
        assert _n_params(ReconstructionNet(depth="mlp3_small")) == PILOT_PARAMS

    def test_mlp3_paper_param_count(self):
        net = ReconstructionNet(depth="mlp3_paper")
        assert _n_params(net) == PAPER_PARAMS
        assert _n_params(net) == _n_params(_source_mlp3())

    def test_mlp3_alias_is_mlp3_small(self):
        a = ReconstructionNet(depth="mlp3")
        assert a.depth == "mlp3_small"
        b = ReconstructionNet(depth="mlp3_small")
        assert list(a.state_dict()) == list(b.state_dict())
        # an old "mlp3" (pilot) state dict loads into mlp3_small
        b.load_state_dict(a.state_dict())

    def test_mlp3_paper_matches_source_layer_by_layer(self):
        ours = list(ReconstructionNet(depth="mlp3_paper").net)
        ref = list(_source_mlp3())
        assert [type(m) for m in ours] == [type(m) for m in ref]
        for o, r in zip(ours, ref):
            if isinstance(o, nn.Linear):
                assert (o.in_features, o.out_features) == (r.in_features, r.out_features)
            if isinstance(o, nn.LayerNorm):
                assert o.normalized_shape == r.normalized_shape
            if isinstance(o, nn.Dropout):
                assert o.p == r.p == 0.1
        # same parameter names/shapes => source weights load directly
        ReconstructionNet(depth="mlp3_paper").net.load_state_dict(_source_mlp3().state_dict())

    def test_layernorm_only_in_paper_variant(self):
        def n_ln(d):
            return sum(isinstance(m, nn.LayerNorm) for m in ReconstructionNet(
                d_in=64, d_out=32, hidden_dim=48, depth=d).modules())
        assert n_ln("mlp3_paper") == 2
        assert n_ln("mlp3_small") == 0

    def test_hidden_dim_clamped_like_source(self):
        net = ReconstructionNet(d_in=16, d_out=8, hidden_dim=999, depth="mlp3_paper")
        assert net.net[4].out_features == 16          # min(999, max(16, 8))

    def test_paper_variant_keeps_pytorch_default_init(self):
        # source does not re-initialise the LLaVA MLP: default nn.Linear init
        # has non-zero biases; the LADP init zeroes them.
        paper = ReconstructionNet(d_in=64, d_out=32, hidden_dim=48, depth="mlp3_paper")
        small = ReconstructionNet(d_in=64, d_out=32, hidden_dim=48, depth="mlp3_small")
        assert paper.net[0].bias.abs().sum() > 0
        assert small.net[0].bias.abs().sum() == 0

    @pytest.mark.parametrize("depth", ["mlp1", "mlp2", "mlp3_small", "mlp3_paper"])
    def test_output_shape_and_finite(self, depth):
        net = ReconstructionNet(d_in=64, d_out=32, hidden_dim=48, depth=depth).eval()
        out = net(torch.randn(7, 64))
        assert out.shape == (7, 32)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("depth", ["mlp3_small", "mlp3_paper"])
    def test_save_load_roundtrip(self, depth, tmp_path):
        a = ReconstructionNet(d_in=64, d_out=32, hidden_dim=48, depth=depth).eval()
        torch.save(a.state_dict(), tmp_path / "w.pt")
        b = ReconstructionNet(d_in=64, d_out=32, hidden_dim=48, depth=depth)
        b.load_state_dict(torch.load(tmp_path / "w.pt", weights_only=True))
        b.eval()
        x = torch.randn(5, 64)
        assert torch.equal(a(x), b(x))

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError):
            ReconstructionNet(depth="mlp4")


# ---------------------------------------------------------------------------
# Trainer helpers
# ---------------------------------------------------------------------------

DV, DT = 32, 64


def _write_cache(d, n_files=8, n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    for i in range(n_files):
        torch.save({"E": torch.randn(n, DV, generator=g), "T": torch.randn(n, DT, generator=g)},
                   os.path.join(d, f"features_{i:06d}.pt"))


def _cfg(ckpt_root, **training):
    t = {"epochs": 4, "batch_size": 16, "patience": 5, "val_fraction": 0.25, "lr": 1e-3,
         "files_per_shard": 3, "seed": 42}
    t.update(training)
    return {"checkpoints_dir": str(ckpt_root), "training": t}


def _net(depth="mlp3_paper", dropout=0.1):
    torch.manual_seed(123)
    return ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=48, depth=depth, dropout=dropout)


PAPER_TRAINING = dict(optimizer="adamw", scheduler="warmup_cosine", warmup_steps=5,
                      min_lr_ratio=0.01, grad_clip=1.0, loss_reduction="mean",
                      patience=10, min_delta=1e-3)


# ---------------------------------------------------------------------------
# Optimiser / schedule / clipping
# ---------------------------------------------------------------------------

class TestOptimisation:
    def test_defaults_are_original_ladp(self, tmp_path):
        t = ReconstructionTrainer(_net("mlp3_small"), _cfg(tmp_path))
        assert (t.optimizer_name, t.scheduler_name, t.grad_clip, t.min_delta, t.loss_reduction) == \
               ("adam", "none", None, 0.0, "sum")
        assert type(t._make_optimizer()) is torch.optim.Adam

    def test_adamw_uses_torch_default_weight_decay(self, tmp_path):
        t = ReconstructionTrainer(_net(), _cfg(tmp_path, optimizer="adamw"))
        opt = t._make_optimizer()
        assert type(opt) is torch.optim.AdamW
        assert opt.param_groups[0]["weight_decay"] == 0.01

    def test_schedule_matches_source_sequential_lr(self, tmp_path):
        lr, warm, total = 1e-4, 20, 200
        t = ReconstructionTrainer(_net(), _cfg(tmp_path, lr=lr, scheduler="warmup_cosine",
                                               warmup_steps=warm, min_lr_ratio=0.01))
        p1 = [nn.Parameter(torch.zeros(1))]
        p2 = [nn.Parameter(torch.zeros(1))]
        o1 = torch.optim.AdamW(p1, lr=lr)
        o2 = torch.optim.AdamW(p2, lr=lr)
        ours = t._make_scheduler(o1, total)
        ref = _source_scheduler(o2, lr, warm, total)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step in range(total):
                assert o1.param_groups[0]["lr"] == pytest.approx(o2.param_groups[0]["lr"], rel=1e-6, abs=1e-12), step
                o1.step(); o2.step(); ours.step(); ref.step()
        assert o1.param_groups[0]["lr"] == pytest.approx(0.01 * lr)

    def test_schedule_clamps_after_total(self, tmp_path):
        t = ReconstructionTrainer(_net(), _cfg(tmp_path, scheduler="warmup_cosine",
                                               warmup_steps=2, min_lr_ratio=0.01))
        o = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0)
        s = t._make_scheduler(o, 10)
        for _ in range(30):
            o.step(); s.step()
        assert o.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_grad_clip_bounds_gradient_norm(self, tmp_path):
        t = ReconstructionTrainer(_net(), _cfg(tmp_path, grad_clip=1e-3, loss_reduction="sum"))
        t._E_mean, t._E_std_v = torch.zeros(DV), torch.ones(DV)
        t._T_mean, t._T_std_v = torch.zeros(DT), torch.ones(DT)
        norms = []

        class Rec(torch.optim.SGD):
            def step(self, closure=None):
                g = torch.cat([p.grad.flatten() for grp in self.param_groups for p in grp["params"]])
                norms.append(g.norm().item())
                return super().step(closure)

        opt = Rec(t.recon_net.parameters(), lr=0.0)
        batches = [(torch.randn(16, DV), torch.randn(16, DT)) for _ in range(3)]
        t._train_epoch(batches, opt)
        assert norms and max(norms) <= 1e-3 * (1 + 1e-4)

    def test_reported_train_loss_independent_of_reduction(self, tmp_path):
        out = {}
        batches = [(torch.randn(16, DV), torch.randn(16, DT)) for _ in range(3)]
        for red in ("sum", "mean"):
            t = ReconstructionTrainer(_net(dropout=0.0), _cfg(tmp_path, loss_reduction=red))
            t._E_mean, t._E_std_v = torch.zeros(DV), torch.ones(DV)
            t._T_mean, t._T_std_v = torch.zeros(DT), torch.ones(DT)
            out[red] = t._train_epoch(batches, torch.optim.SGD(t.recon_net.parameters(), lr=0.0))
        assert out["sum"] == pytest.approx(out["mean"], rel=1e-5)

    @pytest.mark.parametrize("key,val", [("optimizer", "sgd"), ("scheduler", "step"),
                                         ("loss_reduction", "max")])
    def test_invalid_options_raise(self, tmp_path, key, val):
        with pytest.raises(ValueError):
            ReconstructionTrainer(_net(), _cfg(tmp_path, **{key: val}))


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def _run_with_val_sequence(tmp_path, monkeypatch, vals, **training):
    cache = tmp_path / "cache"; cache.mkdir()
    _write_cache(cache, n_files=4, n=8)
    t = ReconstructionTrainer(_net(dropout=0.0), _cfg(tmp_path / "ck", epochs=len(vals), **training))
    it = iter(vals)
    monkeypatch.setattr(t, "_val_epoch", lambda loader: next(it))
    saves = []
    orig = t._atomic_save
    monkeypatch.setattr(t, "_atomic_save",
                        lambda obj, p: (saves.append(os.path.basename(p)), orig(obj, p)))
    t.fit(str(cache))
    return t, saves


class TestEarlyStopping:
    def test_min_delta_requires_real_improvement(self, tmp_path, monkeypatch):
        # 1.0 is best; later values improve by < 1e-3, so they do not count
        vals = [1.0, 0.9995, 0.9991, 0.9990, 0.5, 0.4]
        t, saves = _run_with_val_sequence(tmp_path, monkeypatch, vals, patience=3, min_delta=1e-3)
        assert len(t.val_loss_history) == 4          # stopped after 3 non-improving epochs
        assert saves.count("best.pt") == 1

    def test_min_delta_zero_counts_any_improvement(self, tmp_path, monkeypatch):
        vals = [1.0, 0.9995, 0.9991, 0.9990]
        t, saves = _run_with_val_sequence(tmp_path, monkeypatch, vals, patience=3, min_delta=0.0)
        assert len(t.val_loss_history) == 4 and saves.count("best.pt") == 4

    def test_patience_zero_stops_at_first_non_improvement(self, tmp_path, monkeypatch):
        vals = [1.0, 0.9, 0.95, 0.8]
        t, _ = _run_with_val_sequence(tmp_path, monkeypatch, vals, patience=0)
        assert len(t.val_loss_history) == 3


# ---------------------------------------------------------------------------
# data.max_images
# ---------------------------------------------------------------------------

class TestMaxImages:
    def _split(self, tmp_path, monkeypatch, max_images):
        cache = tmp_path / "cache"; cache.mkdir(exist_ok=True)
        if not any(cache.iterdir()):
            _write_cache(cache, n_files=12, n=4)
        cfg = _cfg(tmp_path / "ck", epochs=1)
        cfg["data"] = {"max_images": max_images}
        t = ReconstructionTrainer(_net(dropout=0.0), cfg)
        seen = {}
        import loss_estimator as le
        real = le.ShardedPatchLoader

        def spy(files, bs, fps, shuffle, seed, device):
            seen.setdefault("lists", []).append(list(files))
            return real(files, bs, fps, shuffle=shuffle, seed=seed, device=device)

        monkeypatch.setattr(le, "ShardedPatchLoader", spy)
        t.fit(str(cache))
        train, val = seen["lists"][0], seen["lists"][1]
        return train, val

    def test_subset_size_and_disjoint_by_image(self, tmp_path, monkeypatch):
        train, val = self._split(tmp_path, monkeypatch, 8)
        assert len(train) + len(val) == 8
        assert not set(train) & set(val)

    def test_subset_is_deterministic(self, tmp_path, monkeypatch):
        a = self._split(tmp_path, monkeypatch, 8)
        b = self._split(tmp_path, monkeypatch, 8)
        assert a == b

    def test_null_uses_all_images(self, tmp_path, monkeypatch):
        train, val = self._split(tmp_path, monkeypatch, None)
        assert len(train) + len(val) == 12


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _fit(cache, ck, epochs, **kw):
    t = ReconstructionTrainer(_net(), _cfg(ck, epochs=epochs, **PAPER_TRAINING))
    t.fit(str(cache), **kw)
    return t


class TestResume:
    def test_last_pt_resume_reproduces_uninterrupted_run(self, tmp_path):
        cache = tmp_path / "cache"; cache.mkdir()
        _write_cache(cache)
        ref = _fit(cache, tmp_path / "a", 5)
        ref_last = torch.load(tmp_path / "a" / "recon" / "last.pt", weights_only=True)

        # Crash during epoch 3 of a 5-epoch run (same config => same schedule).
        crash = ReconstructionTrainer(_net(), _cfg(tmp_path / "b", epochs=5, **PAPER_TRAINING))
        orig, calls = crash._train_epoch, []

        def crashing(loader, opt):
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return orig(loader, opt)

        crash._train_epoch = crashing
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        assert torch.load(tmp_path / "b" / "recon" / "last.pt", weights_only=True)["epoch"] == 2

        res = _fit(cache, tmp_path / "b", 5,
                   resume_from=str(tmp_path / "b" / "recon" / "last.pt"))
        res_last = torch.load(tmp_path / "b" / "recon" / "last.pt", weights_only=True)

        assert res.train_loss_history == ref.train_loss_history     # all 5 epochs, exact
        assert res.val_loss_history == ref.val_loss_history
        for k in ref_last["model"]:
            assert torch.equal(ref_last["model"][k], res_last["model"][k]), k
        assert res_last["scheduler"]["last_epoch"] == ref_last["scheduler"]["last_epoch"]

    def test_old_format_last_pt_still_loads(self, tmp_path):
        cache = tmp_path / "cache"; cache.mkdir()
        _write_cache(cache)
        t = _fit(cache, tmp_path / "c", 2)
        p = tmp_path / "c" / "recon" / "last.pt"
        ck = torch.load(p, weights_only=True)
        for k in ("scheduler", "history", "rng"):                   # pilot-era format
            ck.pop(k)
        torch.save(ck, p)
        t2 = _fit(cache, tmp_path / "c", 3, resume_from=str(p))
        assert len(t2.val_loss_history) == 1                         # only epoch 3 ran

    def test_weights_only_resume(self, tmp_path):
        cache = tmp_path / "cache"; cache.mkdir()
        _write_cache(cache)
        _fit(cache, tmp_path / "d", 2)
        t = _fit(cache, tmp_path / "d", 4, resume_from=str(tmp_path / "d" / "recon" / "best.pt"),
                 start_epoch=3, best_val_loss=10.0)
        assert len(t.val_loss_history) == 2


# ---------------------------------------------------------------------------
# CLI (scripts/train_recon.py) in a throw-away project
# ---------------------------------------------------------------------------

def _cli_project(root: Path, epochs: int):
    import shutil
    import yaml
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    shutil.copy(ROOT / "src" / "loss_estimator.py", root / "src" / "loss_estimator.py")
    shutil.copy(ROOT / "scripts" / "train_recon.py", root / "scripts" / "train_recon.py")
    c = root / "data" / "cache" / "coco_train"
    c.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    for i in range(6):
        torch.save({"E": torch.randn(8, 1024, generator=g).half(),
                    "T": torch.randn(8, 4096, generator=g).half()}, c / f"features_{i:06d}.pt")
    cfg = {"output_dir": "results/run", "data": {"cache_dir": "data/cache"},
           "model": {"device": "cpu"},
           "loss": {"reconstructor": "mlp2", "hidden_dim": 16, "dropout": 0.1},
           "training": {"epochs": epochs, "lr": 1e-3, "batch_size": 16, "seed": 42,
                        "val_fraction": 0.34, "files_per_shard": 2, **PAPER_TRAINING}}
    (root / "configs" / "c.yaml").write_text(yaml.safe_dump(cfg))


def _cli(root: Path, *extra):
    p = subprocess.run([sys.executable, "scripts/train_recon.py", "--config", "configs/c.yaml", *extra],
                       cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


class TestCLI:
    def test_resume_from_last_pt_keeps_full_history(self, tmp_path):
        import yaml
        _cli_project(tmp_path, epochs=2)
        rc, _ = _cli(tmp_path)
        assert rc == 0
        cfgp = tmp_path / "configs" / "c.yaml"
        cfg = yaml.safe_load(cfgp.read_text()); cfg["training"]["epochs"] = 4
        cfgp.write_text(yaml.safe_dump(cfg))
        rc, out = _cli(tmp_path, "--resume", "checkpoints/recon/last.pt")
        assert rc == 0, out
        assert "Epochs run          : 4 / 4" in out

    def test_start_epoch_without_resume_is_rejected(self, tmp_path):
        _cli_project(tmp_path, epochs=1)
        rc, out = _cli(tmp_path, "--start-epoch", "3")
        assert rc != 0 and "only valid together with --resume" in out

    def test_weights_only_resume_requires_flags(self, tmp_path):
        _cli_project(tmp_path, epochs=1)
        assert _cli(tmp_path)[0] == 0
        rc, out = _cli(tmp_path, "--resume", "checkpoints/recon/best.pt")
        assert rc != 0 and "--start-epoch" in out
