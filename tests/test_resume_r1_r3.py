"""
Regression tests for review-2 findings (docs/AUDIT_code_review_2.md):

R1  Resuming with more epochs must be accepted (also with training.warmup_ratio)
    and the learning-rate curve must stay continuous at the resume point.
R3  best.pt must hold this run's best weights: a weights-only resume makes the
    resumed weights the best (an older best.pt is kept as a backup); a full
    resume refuses a foreign best.pt; the final reload is verified.
"""

import glob
import logging
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import loss_estimator as le  # noqa: E402

FLOOR = 0.01
LR = 1e-3


def _cache(d: Path, n_files=8, n=8):
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for i in range(n_files):
        torch.save({"E": torch.randn(n, 16, generator=g), "T": torch.randn(n, 24, generator=g)},
                   d / f"features_{i:06d}.pt")
    return d


def _trainer(ck, **tr):
    t = dict(epochs=6, batch_size=8, val_fraction=0.25, files_per_shard=2, seed=0, lr=LR,
             optimizer="adamw", scheduler="warmup_cosine", warmup_steps=3, min_lr_ratio=FLOOR,
             grad_clip=1.0, loss_reduction="mean", patience=50, min_delta=1e-3)
    t.update(tr)
    torch.manual_seed(1)
    net = le.ReconstructionNet(d_in=24, d_out=16, hidden_dim=12, depth="mlp2")
    return le.ReconstructionTrainer(net, {"checkpoints_dir": str(ck), "training": t})


def _crash_after(t, n_epochs):
    orig, calls = t._train_epoch, []

    def f(loader, opt):
        calls.append(1)
        if len(calls) > n_epochs:
            raise KeyboardInterrupt
        return orig(loader, opt)
    t._train_epoch = f


def _lr_trace(t):
    """lr before and after every epoch of a run."""
    trace = []
    orig = t._train_epoch

    def f(loader, opt):
        before = opt.param_groups[0]["lr"]
        out = orig(loader, opt)
        trace.append((before, opt.param_groups[0]["lr"]))
        return out
    t._train_epoch = f
    return trace


def _last(ck):
    return torch.load(Path(ck) / "recon" / "last.pt", weights_only=True)


def _sd_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


# ---------------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------------

class TestR1EpochExtension:
    def test_warmup_ratio_extension_is_accepted_and_warmup_frozen(self, tmp_path):
        cache = _cache(tmp_path / "c")
        _trainer(tmp_path / "ck", epochs=4, warmup_ratio=0.25).fit(str(cache))
        w0 = _last(tmp_path / "ck")["schedule"]["warmup_steps"]
        t = _trainer(tmp_path / "ck", epochs=8, warmup_ratio=0.25)          # used to raise ValueError
        t.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
        last = _last(tmp_path / "ck")
        s, spe = last["schedule"], last["run_info"]["steps_per_epoch"]
        assert len(t.val_loss_history) == 8
        assert s["warmup_steps"] == w0 and s["total_steps"] == 8 * spe     # warm-up kept, cosine to new end
        assert last["optimizer"]["param_groups"][0]["lr"] == pytest.approx(FLOOR * LR)

    @pytest.mark.parametrize("extra", [dict(warmup_steps=3), dict(warmup_ratio=0.2)])
    def test_lr_continuous_at_resume_and_reaches_floor(self, tmp_path, extra):
        cache = _cache(tmp_path / "c")
        crash = _trainer(tmp_path / "ck", epochs=6, **extra)
        _crash_after(crash, 3)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        ck = _last(tmp_path / "ck")
        done, lr_at_resume = ck["scheduler"]["last_epoch"], ck["optimizer"]["param_groups"][0]["lr"]
        old = ck["schedule"]
        assert lr_at_resume > 2 * FLOOR * LR                                  # mid-curve, not at the floor

        t = _trainer(tmp_path / "ck", epochs=12, **extra)                     # extend 6 -> 12
        trace = _lr_trace(t)
        t.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
        new = _last(tmp_path / "ck")["schedule"]
        f = le.ReconstructionTrainer._lr_factor
        assert f(new, FLOOR, done) == pytest.approx(f(old, FLOOR, done))      # continuous at the resume step
        assert trace[0][0] == pytest.approx(lr_at_resume)
        lrs = [lr_at_resume] + [a for _, a in trace]
        assert all(b <= a + 1e-15 for a, b in zip(lrs, lrs[1:]))             # monotone decay, no jump up
        assert lrs[-1] == pytest.approx(FLOOR * LR)                            # floor at the new end
        assert new["rebase_step"] == done and new["warmup_steps"] == old["warmup_steps"]

    def test_old_behaviour_would_have_jumped(self, tmp_path):
        """The bug, shown on the schedule maths: recomputing the cosine with the new total jumps up."""
        old = {"warmup_steps": 3, "total_steps": 18, "rebase_step": None, "rebase_factor": None}
        naive = dict(old, total_steps=36)
        f = le.ReconstructionTrainer._lr_factor
        assert f(naive, FLOOR, 12) > 1.5 * f(old, FLOOR, 12)

    def test_chained_extensions_stay_continuous(self, tmp_path):
        cache = _cache(tmp_path / "c")
        crash = _trainer(tmp_path / "ck", epochs=6)
        _crash_after(crash, 2)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        f = le.ReconstructionTrainer._lr_factor
        prev = _last(tmp_path / "ck")
        for epochs, stop in ((9, 4), (12, None)):
            t = _trainer(tmp_path / "ck", epochs=epochs)
            if stop:
                _crash_after(t, stop - prev["epoch"])
                with pytest.raises(KeyboardInterrupt):
                    t.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
            else:
                t.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
            cur = _last(tmp_path / "ck")
            done = prev["scheduler"]["last_epoch"]
            assert f(cur["schedule"], FLOOR, done) == pytest.approx(f(prev["schedule"], FLOOR, done))
            prev = cur
        assert prev["optimizer"]["param_groups"][0]["lr"] == pytest.approx(FLOOR * LR)

    def test_extension_during_warmup_keeps_linear_warmup(self, tmp_path):
        cache = _cache(tmp_path / "c")
        crash = _trainer(tmp_path / "ck", epochs=6, warmup_steps=12)          # 3 steps/epoch -> 4 epochs warm-up
        _crash_after(crash, 1)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        t = _trainer(tmp_path / "ck", epochs=10, warmup_steps=12)
        t.fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
        s = _last(tmp_path / "ck")["schedule"]
        assert s["warmup_steps"] == 12 and s["rebase_step"] is None and len(t.val_loss_history) == 10

    @pytest.mark.parametrize("before,after", [(dict(warmup_ratio=0.2), dict(warmup_ratio=0.3)),
                                              (dict(warmup_steps=3), dict(warmup_steps=5)),
                                              (dict(warmup_steps=3), dict(warmup_ratio=0.2))])
    def test_changing_the_configured_warmup_is_refused(self, tmp_path, before, after):
        cache = _cache(tmp_path / "c")
        _trainer(tmp_path / "ck", epochs=2, **before).fit(str(cache))
        with pytest.raises(ValueError, match="warmup_config"):
            _trainer(tmp_path / "ck", epochs=4, **after).fit(
                str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))

    def test_shrinking_below_completed_steps_is_refused(self, tmp_path):
        cache = _cache(tmp_path / "c")
        crash = _trainer(tmp_path / "ck", epochs=6)
        _crash_after(crash, 4)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        with pytest.raises(ValueError, match="already"):
            _trainer(tmp_path / "ck", epochs=2).fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))

    def test_same_epochs_resume_is_still_exact(self, tmp_path):
        cache = _cache(tmp_path / "c")
        ref = _trainer(tmp_path / "a", epochs=5)
        ref.fit(str(cache))
        crash = _trainer(tmp_path / "b", epochs=5)
        _crash_after(crash, 2)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        res = _trainer(tmp_path / "b", epochs=5)
        res.fit(str(cache), resume_from=str(tmp_path / "b" / "recon" / "last.pt"))
        assert res.val_loss_history == ref.val_loss_history
        assert _sd_equal(_last(tmp_path / "a")["model"], _last(tmp_path / "b")["model"])
        assert _last(tmp_path / "b")["schedule"]["rebase_step"] is None


# ---------------------------------------------------------------------------
# R3
# ---------------------------------------------------------------------------

class TestR3BestCheckpoint:
    def _weights(self, seed):
        torch.manual_seed(seed)
        return le.ReconstructionNet(d_in=24, d_out=16, hidden_dim=12, depth="mlp2").state_dict()

    def test_weights_only_resume_replaces_stale_best_and_keeps_a_backup(self, tmp_path):
        cache = _cache(tmp_path / "c")
        run = tmp_path / "ck" / "recon"
        run.mkdir(parents=True)
        stale, mine = self._weights(7), self._weights(8)
        torch.save(stale, run / "best.pt")
        torch.save(mine, tmp_path / "mine.pt")
        t = _trainer(tmp_path / "ck", epochs=3)
        t._val_epoch = lambda loader: 9.9                                     # nothing beats --best-val
        t.fit(str(cache), resume_from=str(tmp_path / "mine.pt"), start_epoch=2, best_val_loss=0.5)
        assert _sd_equal(t.recon_net.state_dict(), mine)                      # final model = resumed weights
        backups = glob.glob(str(run / "best.superseded_*.pt"))
        assert len(backups) == 1 and _sd_equal(torch.load(backups[0], weights_only=True), stale)
        ns = le.NormStats.load(str(run / "norm_stats.pt"))
        assert ns.meta["model_sha256"] == le.state_dict_fingerprint(mine)    # stats fitted on the right weights

    def test_resume_from_the_folders_own_best_needs_no_backup(self, tmp_path):
        cache = _cache(tmp_path / "c")
        _trainer(tmp_path / "ck", epochs=2).fit(str(cache))
        run = tmp_path / "ck" / "recon"
        t = _trainer(tmp_path / "ck", epochs=4)
        t.fit(str(cache), resume_from=str(run / "best.pt"), start_epoch=3, best_val_loss=10.0)
        assert not glob.glob(str(run / "best.superseded_*.pt"))

    def test_full_resume_refuses_a_foreign_best(self, tmp_path):
        cache = _cache(tmp_path / "c")
        crash = _trainer(tmp_path / "ck", epochs=5)
        _crash_after(crash, 2)
        with pytest.raises(KeyboardInterrupt):
            crash.fit(str(cache))
        run = tmp_path / "ck" / "recon"
        assert _last(tmp_path / "ck")["best_sha256"] == le.ReconstructionTrainer._file_fingerprint(str(run / "best.pt"))
        torch.save(self._weights(9), run / "best.pt")                          # replaced by something else
        t = _trainer(tmp_path / "ck", epochs=5)
        trained = []
        orig = t._train_epoch
        t._train_epoch = lambda l, o: (trained.append(1), orig(l, o))[1]
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            t.fit(str(cache), resume_from=str(run / "last.pt"))
        assert trained == []                                                   # refused before any training

    def test_best_changed_during_the_run_is_caught(self, tmp_path, monkeypatch):
        cache = _cache(tmp_path / "c")
        run = tmp_path / "ck" / "recon"
        t = _trainer(tmp_path / "ck", epochs=2)
        vals = iter([1.0, 2.0])

        def val(loader):
            v = next(vals)
            if v == 2.0:                                                       # someone overwrites best.pt
                torch.save(self._weights(11), run / "best.pt")
            return v
        monkeypatch.setattr(t, "_val_epoch", val)
        with pytest.raises(RuntimeError, match="changed during the run"):
            t.fit(str(cache))

    def test_old_checkpoint_without_best_record_still_resumes(self, tmp_path, caplog):
        cache = _cache(tmp_path / "c")
        _trainer(tmp_path / "ck", epochs=2).fit(str(cache))
        p = tmp_path / "ck" / "recon" / "last.pt"
        ck = torch.load(p, weights_only=True)
        ck.pop("best_sha256")
        ck.pop("schedule")
        torch.save(ck, p)
        with caplog.at_level(logging.WARNING, logger="loss_estimator"):
            t = _trainer(tmp_path / "ck", epochs=3)
            t.fit(str(cache), resume_from=str(p))
        assert "older format" in caplog.text and len(t.val_loss_history) == 3
