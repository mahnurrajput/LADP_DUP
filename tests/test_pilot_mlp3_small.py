"""
Tests of the FYP-1 pilot reconstruction module (``mlp3``, 14,685,184 params),
written against the pilot-era API (commit 9a488d8) so they also run on the
branch that holds exactly that code.

* architecture: layer sizes, parameter count, (576, 4096) -> (576, 1024)
* training + resume: full resume from last.pt and weights-only resume from
  best.pt (--start-epoch / --best-val), as used to resume the pilot after its
  crash at epoch 7
* the trained pilot checkpoint loads (skipped if absent; set LADP_DATA_ROOT to
  the repository folder that holds checkpoints/ when running from elsewhere)
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loss_estimator import NormStats, ReconstructionNet, ReconstructionTrainer  # noqa: E402

DATA_ROOT = Path(os.environ.get("LADP_DATA_ROOT", ROOT))
PILOT = next((p for p in (DATA_ROOT / "checkpoints" / "recon_pilot", DATA_ROOT / "checkpoints" / "recon")
              if (p / "best.pt").exists()), None)


def test_architecture_and_parameter_count():
    net = ReconstructionNet(depth="mlp3")
    lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin] == [(4096, 2048), (2048, 2048), (2048, 1024)]
    assert net.param_count() == 14_685_184
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in net.modules())


def test_forward_shape():
    net = ReconstructionNet(depth="mlp3").eval()
    with torch.no_grad():
        out = net(torch.randn(576, 4096))
    assert out.shape == (576, 1024) and torch.isfinite(out).all()


def _cache(d: Path, n_files=6, n=8):
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for i in range(n_files):
        torch.save({"E": torch.randn(n, 16, generator=g).half(), "T": torch.randn(n, 24, generator=g).half()},
                   d / f"features_{i:06d}.pt")
    return d


def _trainer(ck, epochs):
    torch.manual_seed(0)
    net = ReconstructionNet(d_in=24, d_out=16, hidden_dim=12, depth="mlp3")
    cfg = {"checkpoints_dir": str(ck),
           "training": {"epochs": epochs, "lr": 1e-3, "batch_size": 16, "patience": 5,
                        "val_fraction": 0.34, "seed": 42, "files_per_shard": 2}}
    return ReconstructionTrainer(net, cfg)


def test_training_writes_checkpoints_and_norm_stats(tmp_path):
    cache = _cache(tmp_path / "c")
    _, ns = _trainer(tmp_path / "ck", 2).fit(str(cache))
    run = tmp_path / "ck" / "recon"
    last = torch.load(run / "last.pt", weights_only=True)
    assert {"model", "optimizer", "epoch", "best_val_loss", "patience_count"} <= set(last)
    assert last["epoch"] == 2 and (run / "best.pt").exists() and (run / "norm_stats.pt").exists()
    assert ns.E_mean.shape == (16,) and ns.L_p5 <= ns.L_p95


def test_full_resume_from_last_pt_continues(tmp_path):
    cache = _cache(tmp_path / "c")
    _trainer(tmp_path / "ck", 2).fit(str(cache))
    _trainer(tmp_path / "ck", 4).fit(str(cache), resume_from=str(tmp_path / "ck" / "recon" / "last.pt"))
    assert torch.load(tmp_path / "ck" / "recon" / "last.pt", weights_only=True)["epoch"] == 4


def test_weights_only_resume_from_best_pt(tmp_path):
    """The pilot was resumed this way: --resume best.pt --start-epoch 8 --best-val 0.076641."""
    cache = _cache(tmp_path / "c")
    _trainer(tmp_path / "ck", 2).fit(str(cache))
    run = tmp_path / "ck" / "recon"
    _trainer(tmp_path / "ck", 4).fit(str(cache), resume_from=str(run / "best.pt"),
                                     start_epoch=3, best_val_loss=10.0)
    assert torch.load(run / "last.pt", weights_only=True)["epoch"] == 4


@pytest.mark.skipif(PILOT is None, reason="pilot checkpoint not present")
def test_trained_pilot_checkpoint_loads():
    net = ReconstructionNet(depth="mlp3")
    net.load_state_dict(torch.load(PILOT / "best.pt", map_location="cpu", weights_only=True))
    ns = NormStats.load(str(PILOT / "norm_stats.pt"))
    assert ns.E_mean.shape == (1024,) and ns.T_mean.shape == (4096,)
    assert ns.L_p5 == pytest.approx(0.020985, abs=1e-6) and ns.L_p95 == pytest.approx(0.142976, abs=1e-6)
