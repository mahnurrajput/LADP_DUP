"""
Tests for the memory-efficient training data path in loss_estimator.py:
ShardedPatchLoader, streaming normalisation statistics, image-level
train/val split and p5/p95 computation.

Each synthetic patch carries a unique integer id in E[:, 0] (and the id of
its image in E[:, 1]) so we can check coverage and mixing exactly.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import loss_estimator as le  # noqa: E402
from loss_estimator import ReconstructionNet, ReconstructionTrainer, ShardedPatchLoader  # noqa: E402

DV, DT = 8, 16


def _write_files(tmp_path, n_files=10, n_patches=12, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    paths = []
    pid = 0
    for i in range(n_files):
        E = torch.randn(n_patches, DV, generator=g) * 2 + 1
        T = torch.randn(n_patches, DT, generator=g) * 3 - 1
        E[:, 0] = torch.arange(pid, pid + n_patches, dtype=torch.float32)
        E[:, 1] = i
        pid += n_patches
        p = tmp_path / f"features_{i:06d}.pt"
        torch.save({"E": E.to(dtype), "T": T.to(dtype)}, p)
        paths.append(str(p))
    return paths


def _ids(loader):
    out = []
    for e, _ in loader:
        out.extend(e[:, 0].long().tolist())
    return out


# ---------------------------------------------------------------------------
# Coverage / batching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("files_per_shard,batch_size", [(1, 5), (3, 7), (4, 48), (100, 1000)])
def test_every_patch_exactly_once_per_epoch(tmp_path, files_per_shard, batch_size):
    paths = _write_files(tmp_path, n_files=10, n_patches=12)
    loader = ShardedPatchLoader(paths, batch_size, files_per_shard, shuffle=True, seed=1)
    for _ in range(2):  # two epochs
        ids = _ids(loader)
        assert sorted(ids) == list(range(120))


def test_full_576_patches_per_image_are_yielded(tmp_path):
    paths = _write_files(tmp_path, n_files=3, n_patches=576)
    loader = ShardedPatchLoader(paths, 128, files_per_shard=2, shuffle=True)
    imgs = []
    for e, _ in loader:
        imgs.extend(e[:, 1].long().tolist())
    for i in range(3):
        assert imgs.count(i) == 576


def test_batch_sizes(tmp_path):
    paths = _write_files(tmp_path, n_files=4, n_patches=10)  # 40 patches, 1 shard
    sizes = [e.shape[0] for e, _ in ShardedPatchLoader(paths, 16, 4, shuffle=False)]
    assert sizes == [16, 16, 8]


def test_e_t_rows_stay_paired(tmp_path):
    paths = _write_files(tmp_path, n_files=5, n_patches=6)
    ref = {}
    for p in paths:
        d = torch.load(p, weights_only=True)
        for e, t in zip(d["E"], d["T"]):
            ref[int(e[0])] = t
    for e, t in ShardedPatchLoader(paths, 7, 2, shuffle=True, seed=3):
        for ei, ti in zip(e, t):
            assert torch.equal(ref[int(ei[0])], ti)


def test_dtype_preserved_fp16(tmp_path):
    paths = _write_files(tmp_path, n_files=2, n_patches=4, dtype=torch.float16)
    e, t = next(iter(ShardedPatchLoader(paths, 8, 2)))
    assert e.dtype == torch.float16 and t.dtype == torch.float16


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

def test_unshuffled_order_is_file_order(tmp_path):
    paths = _write_files(tmp_path, n_files=6, n_patches=5)
    assert _ids(ShardedPatchLoader(paths, 4, 2, shuffle=False)) == list(range(30))


def test_batches_mix_patches_from_different_images(tmp_path):
    paths = _write_files(tmp_path, n_files=8, n_patches=50)
    loader = ShardedPatchLoader(paths, 32, files_per_shard=8, shuffle=True, seed=0)
    n_imgs = [len(set(e[:, 1].long().tolist())) for e, _ in loader]
    # Unshuffled, a 32-patch batch spans at most 2 images; shuffled it spans many.
    assert min(n_imgs[:-1]) >= 4


def test_reshuffles_each_epoch_and_is_seed_deterministic(tmp_path):
    paths = _write_files(tmp_path, n_files=6, n_patches=10)
    a = ShardedPatchLoader(paths, 8, 3, shuffle=True, seed=7)
    ep1, ep2 = _ids(a), _ids(a)
    assert ep1 != ep2
    b = ShardedPatchLoader(paths, 8, 3, shuffle=True, seed=7)
    assert _ids(b) == ep1 and _ids(b) == ep2


def test_file_order_is_shuffled_across_shards(tmp_path):
    paths = _write_files(tmp_path, n_files=20, n_patches=3)
    loader = ShardedPatchLoader(paths, 1000, files_per_shard=5, shuffle=True, seed=0)
    first_shard_imgs = set(next(iter(loader))[0][:, 1].long().tolist())
    assert first_shard_imgs != {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# I/O and memory behaviour
# ---------------------------------------------------------------------------

def test_each_file_loaded_once_per_epoch(tmp_path, monkeypatch):
    paths = _write_files(tmp_path, n_files=9, n_patches=11)
    calls = []
    real_load = torch.load

    def counting_load(f, *a, **k):
        calls.append(str(f))
        return real_load(f, *a, **k)

    monkeypatch.setattr(le.torch, "load", counting_load)
    loader = ShardedPatchLoader(paths, 4, files_per_shard=2, shuffle=True)
    for epoch in range(2):
        calls.clear()
        _ids(loader)
        assert sorted(calls) == sorted(paths), f"epoch {epoch}"


def test_resident_data_bounded_by_shard_size(tmp_path, monkeypatch):
    paths = _write_files(tmp_path, n_files=12, n_patches=10)
    rows = []
    real = ShardedPatchLoader._load_shard

    def recording(ps):
        E, T = real(ps)
        rows.append(E.shape[0])
        return E, T

    monkeypatch.setattr(ShardedPatchLoader, "_load_shard", staticmethod(recording))
    _ids(ShardedPatchLoader(paths, 4, files_per_shard=3, shuffle=True))
    assert len(rows) == 4 and max(rows) == 3 * 10


def test_empty_file_list_raises():
    with pytest.raises(ValueError):
        ShardedPatchLoader([], 4)


# ---------------------------------------------------------------------------
# Streaming statistics / percentiles
# ---------------------------------------------------------------------------

def _trainer(tmp_path, **training):
    cfg = {"checkpoints_dir": str(tmp_path / "ckpt"),
           "training": {"epochs": 1, "lr": 1e-3, "batch_size": 16, "patience": 1,
                        "val_fraction": 0.2, "seed": 0, **training}}
    net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=8, depth="mlp1")
    return ReconstructionTrainer(net, cfg, device="cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_streaming_mean_std_matches_full_tensor(tmp_path, dtype):
    paths = _write_files(tmp_path, n_files=7, n_patches=13, dtype=dtype)
    E = torch.cat([torch.load(p, weights_only=True)["E"] for p in paths]).double()
    T = torch.cat([torch.load(p, weights_only=True)["T"] for p in paths]).double()
    loader = ShardedPatchLoader(paths, 10, files_per_shard=2, shuffle=False)
    Em, Es, Tm, Ts, n = _trainer(tmp_path)._streaming_mean_std(loader)
    assert n == E.shape[0]
    torch.testing.assert_close(Em, E.mean(0).float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(Es, E.std(0).float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(Tm, T.mean(0).float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(Ts, T.std(0).float(), rtol=1e-5, atol=1e-5)


def test_streaming_std_constant_column_is_clamped(tmp_path):
    p = tmp_path / "features_000000.pt"
    torch.save({"E": torch.ones(5, DV), "T": torch.ones(5, DT)}, p)
    _, Es, _, Ts, _ = _trainer(tmp_path)._streaming_mean_std(
        ShardedPatchLoader([str(p)], 2, 1, shuffle=False))
    assert (Es > 0).all() and (Ts > 0).all()


def test_p5_p95_close_to_torch_quantile(tmp_path):
    tr = _trainer(tmp_path)
    tr._E_mean = tr._E_std_v = torch.zeros(DV)
    tr._T_mean = tr._T_std_v = torch.zeros(DT)
    L = torch.rand(10_001, generator=torch.Generator().manual_seed(0))
    ns = tr._compute_norm_stats(L)
    assert abs(ns.L_p5 - torch.quantile(L, 0.05).item()) < 1e-3
    assert abs(ns.L_p95 - torch.quantile(L, 0.95).item()) < 1e-3
    assert ns.L_p5 < ns.L_p95


# ---------------------------------------------------------------------------
# Image-level split inside fit()
# ---------------------------------------------------------------------------

def test_fit_uses_disjoint_image_level_split(tmp_path, monkeypatch):
    paths = _write_files(tmp_path, n_files=10, n_patches=6)
    seen = []
    Real = le.ShardedPatchLoader

    class Spy(Real):
        def __init__(self, file_paths, *a, **k):
            seen.append((list(file_paths), k.get("shuffle")))
            super().__init__(file_paths, *a, **k)

    monkeypatch.setattr(le, "ShardedPatchLoader", Spy)
    _, ns = _trainer(tmp_path).fit(str(tmp_path))
    train_files, val_files = set(seen[0][0]), set(seen[1][0])
    assert train_files.isdisjoint(val_files)
    assert train_files | val_files == set(paths)
    assert len(val_files) == 2                      # 20% of 10 images
    assert set(seen[2][0]) == train_files           # stats pass = train only
    assert ns.L_p5 <= ns.L_p95
    assert os.path.exists(tmp_path / "ckpt" / "recon" / "best.pt")


def test_fit_requires_two_images(tmp_path):
    _write_files(tmp_path, n_files=1, n_patches=4)
    with pytest.raises(ValueError):
        _trainer(tmp_path).fit(str(tmp_path))
