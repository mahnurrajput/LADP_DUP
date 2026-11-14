"""
test_loss_estimator.py — LADP Stage 1 unit tests
=================================================
Pytest suite for ReconstructionNet, LossEstimator, and NormStats.

All tests use CPU tensors only — no GPU, no real model weights.

Run:
    pytest tests/test_loss_estimator.py -v
"""

import pytest
import torch
import tempfile
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Path setup — make sure we import from ladp/src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loss_estimator import ReconstructionNet, LossEstimator, NormStats, ReconstructionTrainer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Standard dimensions used throughout.
DV = 1024    # pre-connector CLIP feature dim
DT = 4096    # post-connector LLM token dim
HIDDEN = 128  # small hidden dim for fast tests
B = 16        # default batch size


def _make_norm_stats(dv: int = DV, dt: int = DT, l_p5: float = 0.1, l_p95: float = 0.9) -> NormStats:
    """Return a NormStats with zero mean and unit std (identity standardisation)."""
    return NormStats(
        E_mean=torch.zeros(dv),
        E_std=torch.ones(dv),
        T_mean=torch.zeros(dt),
        T_std=torch.ones(dt),
        L_p5=l_p5,
        L_p95=l_p95,
    )


def _make_net(depth: str = "mlp3", dv: int = DV, dt: int = DT) -> ReconstructionNet:
    return ReconstructionNet(d_in=dt, d_out=dv, hidden_dim=HIDDEN, depth=depth, dropout=0.0)


# ===========================================================================
# TestReconstructionNet
# ===========================================================================

class TestReconstructionNet:
    """Tests for ReconstructionNet (gφ) architecture."""

    def test_mlp3_output_shape(self):
        """mlp3 (default): (B, DT) → (B, DV)."""
        net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=HIDDEN, depth="mlp3")
        x = torch.randn(B, DT)
        out = net(x)
        assert out.shape == (B, DV), f"Expected ({B}, {DV}), got {out.shape}"

    def test_mlp1_output_shape(self):
        """mlp1 (linear only): (B, DT) → (B, DV)."""
        net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=HIDDEN, depth="mlp1")
        x = torch.randn(B, DT)
        out = net(x)
        assert out.shape == (B, DV), f"Expected ({B}, {DV}), got {out.shape}"

    def test_mlp2_output_shape(self):
        """mlp2 (one hidden layer): (B, DT) → (B, DV)."""
        net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=HIDDEN, depth="mlp2")
        x = torch.randn(B, DT)
        out = net(x)
        assert out.shape == (B, DV), f"Expected ({B}, {DV}), got {out.shape}"

    def test_output_is_finite(self):
        """Network outputs must be finite (no NaN or Inf) on random input."""
        for depth in ("mlp1", "mlp2", "mlp3"):
            net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=HIDDEN, depth=depth)
            x = torch.randn(B, DT)
            out = net(x)
            assert torch.isfinite(out).all(), (
                f"Non-finite values in {depth} output: {out[~torch.isfinite(out)][:5]}"
            )

    def test_different_batch_sizes(self):
        """Output shape must be (B, DV) for B in {1, 32, 128}."""
        net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=HIDDEN, depth="mlp3")
        net.eval()
        for batch in (1, 32, 128):
            x = torch.randn(batch, DT)
            out = net(x)
            assert out.shape == (batch, DV), (
                f"B={batch}: expected ({batch}, {DV}), got {out.shape}"
            )

    def test_invalid_depth_raises(self):
        """Passing an unknown depth string must raise ValueError."""
        with pytest.raises(ValueError, match="depth must be"):
            ReconstructionNet(d_in=DT, d_out=DV, depth="mlp99")

    def test_param_count_mlp1_less_than_mlp3(self):
        """mlp1 has fewer parameters than mlp3 when hidden_dim is wide enough.

        For the comparison to be valid, hidden_dim must satisfy:
            mlp3 params = d_in*H + H*H + H*d_out  (two hidden layers)
            mlp1 params = d_in*d_out               (single linear)
        We choose d_in=d_out=H=64 so that:
            mlp3: 64*64 + 64*64 + 64*64 = 12,288
            mlp1: 64*64                 =  4,096
        giving mlp1 < mlp3 unambiguously.
        """
        dim = 64
        net1 = ReconstructionNet(d_in=dim, d_out=dim, hidden_dim=dim, depth="mlp1")
        net3 = ReconstructionNet(d_in=dim, d_out=dim, hidden_dim=dim, depth="mlp3")
        assert net1.param_count() < net3.param_count(), (
            f"mlp1 params={net1.param_count()} should be < mlp3 params={net3.param_count()}"
        )

    def test_eval_mode_disables_dropout(self):
        """In eval mode two forward passes with the same input must be identical."""
        net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=HIDDEN, depth="mlp3", dropout=0.5)
        net.eval()
        x = torch.randn(B, DT)
        out1 = net(x)
        out2 = net(x)
        assert torch.allclose(out1, out2), "eval() should produce deterministic output"


# ===========================================================================
# TestLossEstimator
# ===========================================================================

class TestLossEstimator:
    """Tests for LossEstimator (all three distance modes)."""

    def _estimator(self, distance: str = "mse") -> LossEstimator:
        net = _make_net()
        net.eval()
        ns = _make_norm_stats()
        return LossEstimator(net, ns, distance=distance)

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    def test_mse_output_shape(self):
        """compute() with 'mse' distance must return Tensor of shape (N,)."""
        est = self._estimator("mse")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L = est.compute(E, T)
        assert L.shape == (B,), f"Expected ({B},), got {L.shape}"

    def test_norm_relative_output_shape(self):
        """compute() with 'norm_relative' distance must return Tensor of shape (N,)."""
        est = self._estimator("norm_relative")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L = est.compute(E, T)
        assert L.shape == (B,), f"Expected ({B},), got {L.shape}"

    def test_cosine_output_shape(self):
        """compute() with 'cosine' distance must return Tensor of shape (N,)."""
        est = self._estimator("cosine")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L = est.compute(E, T)
        assert L.shape == (B,), f"Expected ({B},), got {L.shape}"

    # ------------------------------------------------------------------
    # Non-negativity
    # ------------------------------------------------------------------

    def test_mse_is_nonnegative(self):
        """MSE loss must be >= 0 for all patches."""
        est = self._estimator("mse")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L = est.compute(E, T)
        assert (L >= 0).all(), f"MSE loss has negative values: {L[L < 0]}"

    def test_norm_relative_is_nonnegative(self):
        """Norm-relative loss must be >= 0 for all patches."""
        est = self._estimator("norm_relative")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L = est.compute(E, T)
        assert (L >= 0).all(), f"norm_relative loss has negative values: {L[L < 0]}"

    def test_cosine_is_bounded(self):
        """Cosine loss (1 - cos_sim) must be in [0, 2] for all patches."""
        est = self._estimator("cosine")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L = est.compute(E, T)
        assert (L >= -1e-6).all(), "Cosine loss has values below 0"
        assert (L <= 2.0 + 1e-6).all(), "Cosine loss has values above 2"

    # ------------------------------------------------------------------
    # Perfect reconstruction → zero loss
    # ------------------------------------------------------------------

    def test_zero_reconstruction_error_gives_zero_loss(self):
        """
        If gφ perfectly reconstructs E (we patch it to return E_std),
        then the MSE loss should be (nearly) zero for every patch.
        """
        # Build a net where the reconstruction is perfect in standardised space.
        # We achieve this by overriding the net's forward method so it returns
        # its input (identity in T_std space) and setting norm stats so that
        # E_std == T_std after standardisation.
        #
        # Concretely: set E_mean = T_mean = 0, E_std = T_std = 1 so the
        # standardised E and T are identical, then give the net an identity
        # mapping (d_in = d_out, single linear layer initialised to I).

        dim = 64  # small square dim so identity is easy
        ns = NormStats(
            E_mean=torch.zeros(dim),
            E_std=torch.ones(dim),
            T_mean=torch.zeros(dim),
            T_std=torch.ones(dim),
            L_p5=0.0,
            L_p95=1.0,
        )

        # Create an exact identity linear layer.
        net = ReconstructionNet(d_in=dim, d_out=dim, depth="mlp1")
        with torch.no_grad():
            net.net[0].weight.copy_(torch.eye(dim))
            net.net[0].bias.fill_(0.0)
        net.eval()

        est = LossEstimator(net, ns, distance="mse")

        # E and T are the same tensor → standardised versions are equal →
        # E_hat = I @ T_std = T_std = E_std → loss ≈ 0.
        X = torch.randn(B, dim)
        L = est.compute(X, X)  # E = T = X

        assert L.shape == (B,)
        assert (L < 1e-10).all(), (
            f"Expected near-zero loss for perfect reconstruction, got max={L.max().item():.2e}"
        )

    # ------------------------------------------------------------------
    # Normalised output range
    # ------------------------------------------------------------------

    def test_normalised_robust_range(self):
        """compute_normalised(..., norm_mode='robust') must return values in [0, 1]."""
        # Use p5=0 and p95=1 so any reasonable loss falls within range.
        ns = _make_norm_stats(l_p5=0.0, l_p95=100.0)
        net = _make_net()
        net.eval()
        est = LossEstimator(net, ns, distance="mse")

        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L_bar = est.compute_normalised(E, T, norm_mode="robust")

        assert L_bar.shape == (B,)
        assert (L_bar >= 0.0).all(), f"L_bar has values < 0: {L_bar[L_bar < 0]}"
        assert (L_bar <= 1.0).all(), f"L_bar has values > 1: {L_bar[L_bar > 1]}"

    def test_normalised_per_image_range(self):
        """compute_normalised(..., norm_mode='per_image') must return values in [0, 1]."""
        est = self._estimator("mse")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        L_bar = est.compute_normalised(E, T, norm_mode="per_image")

        assert L_bar.shape == (B,)
        assert (L_bar >= 0.0).all(), f"L_bar has values < 0: {L_bar[L_bar < 0]}"
        assert (L_bar <= 1.0 + 1e-6).all(), f"L_bar has values > 1: {L_bar[L_bar > 1 + 1e-6]}"

    def test_handles_all_equal_loss(self):
        """
        Edge case: all patches have the same reconstruction error.
        per_image normalisation should not divide by zero; output should be
        all zeros (or uniform, clamped to [0, 1]).
        """
        dim = 32
        ns = NormStats(
            E_mean=torch.zeros(dim),
            E_std=torch.ones(dim),
            T_mean=torch.zeros(dim),
            T_std=torch.ones(dim),
            L_p5=0.0,
            L_p95=1.0,
        )

        # Build a net with a fixed output (same for every input).
        net = ReconstructionNet(d_in=dim, d_out=dim, depth="mlp1")
        with torch.no_grad():
            net.net[0].weight.zero_()      # zero weight → output = bias
            net.net[0].bias.fill_(0.5)     # constant output = 0.5 everywhere
        net.eval()

        est = LossEstimator(net, ns, distance="mse")
        E = torch.randn(B, dim)   # different E per patch
        T = torch.randn(B, dim)   # different T per patch (but output is constant)
        L_bar = est.compute_normalised(E, T, norm_mode="per_image")

        assert L_bar.shape == (B,)
        assert torch.isfinite(L_bar).all(), "per_image normalisation produced NaN/Inf"
        assert (L_bar >= 0.0).all()
        assert (L_bar <= 1.0 + 1e-6).all()

    def test_invalid_distance_raises(self):
        """Passing an unknown distance string must raise ValueError."""
        net = _make_net()
        ns = _make_norm_stats()
        with pytest.raises(ValueError, match="distance must be"):
            LossEstimator(net, ns, distance="euclidean")

    def test_invalid_norm_mode_raises(self):
        """Passing an unknown norm_mode must raise ValueError."""
        est = self._estimator("mse")
        E = torch.randn(B, DV)
        T = torch.randn(B, DT)
        with pytest.raises(ValueError, match="norm_mode must be"):
            est.compute_normalised(E, T, norm_mode="zscore")

    def test_compute_no_grad(self):
        """compute() must not build a gradient graph (runs under no_grad)."""
        est = self._estimator("mse")
        E = torch.randn(B, DV, requires_grad=True)
        T = torch.randn(B, DT)
        L = est.compute(E, T)
        assert not L.requires_grad, "Loss tensor should not require grad"

    def test_single_patch(self):
        """compute() must work for N=1 (a single patch)."""
        est = self._estimator("mse")
        E = torch.randn(1, DV)
        T = torch.randn(1, DT)
        L = est.compute(E, T)
        assert L.shape == (1,)
        assert torch.isfinite(L).all()


# ===========================================================================
# TestNormStats
# ===========================================================================

class TestNormStats:
    """Tests for NormStats save / load round-trip."""

    def _make(self, dv: int = DV, dt: int = DT) -> NormStats:
        return NormStats(
            E_mean=torch.randn(dv),
            E_std=torch.rand(dv) + 0.1,   # positive
            T_mean=torch.randn(dt),
            T_std=torch.rand(dt) + 0.1,
            L_p5=0.05,
            L_p95=0.95,
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        """NormStats saved to disk must load without errors."""
        ns = self._make()
        path = str(tmp_path / "norm_stats.pt")
        ns.save(path)
        ns2 = NormStats.load(path)
        assert isinstance(ns2, NormStats)

    def test_values_preserved(self, tmp_path):
        """All tensor values and scalar fields must survive a save/load cycle."""
        ns = self._make()
        path = str(tmp_path / "norm_stats.pt")
        ns.save(path)
        ns2 = NormStats.load(path)

        assert torch.allclose(ns2.E_mean, ns.E_mean), "E_mean mismatch after roundtrip"
        assert torch.allclose(ns2.E_std, ns.E_std),   "E_std mismatch after roundtrip"
        assert torch.allclose(ns2.T_mean, ns.T_mean), "T_mean mismatch after roundtrip"
        assert torch.allclose(ns2.T_std, ns.T_std),   "T_std mismatch after roundtrip"
        assert ns2.L_p5  == pytest.approx(ns.L_p5,  rel=1e-6), "L_p5 mismatch"
        assert ns2.L_p95 == pytest.approx(ns.L_p95, rel=1e-6), "L_p95 mismatch"

    def test_shapes_preserved(self, tmp_path):
        """Tensor shapes must be preserved across save/load."""
        ns = self._make()
        path = str(tmp_path / "norm_stats.pt")
        ns.save(path)
        ns2 = NormStats.load(path)

        assert ns2.E_mean.shape == (DV,), f"E_mean shape mismatch: {ns2.E_mean.shape}"
        assert ns2.E_std.shape  == (DV,), f"E_std shape mismatch: {ns2.E_std.shape}"
        assert ns2.T_mean.shape == (DT,), f"T_mean shape mismatch: {ns2.T_mean.shape}"
        assert ns2.T_std.shape  == (DT,), f"T_std shape mismatch: {ns2.T_std.shape}"

    def test_load_maps_to_cpu(self, tmp_path):
        """NormStats.load must return tensors on CPU regardless of origin device."""
        ns = self._make()
        path = str(tmp_path / "norm_stats.pt")
        ns.save(path)
        ns2 = NormStats.load(path)

        assert ns2.E_mean.device.type == "cpu"
        assert ns2.T_mean.device.type == "cpu"

    def test_save_creates_parent_dir(self, tmp_path):
        """save() must create intermediate directories if they do not exist."""
        ns = self._make()
        nested_path = str(tmp_path / "deep" / "subdir" / "norm_stats.pt")
        ns.save(nested_path)
        assert Path(nested_path).exists(), "File was not created at nested path"

    def test_l_p5_leq_l_p95(self):
        """L_p5 must be <= L_p95 (basic sanity on the percentile ordering)."""
        ns = self._make()
        assert ns.L_p5 <= ns.L_p95, (
            f"L_p5={ns.L_p5} should be <= L_p95={ns.L_p95}"
        )
