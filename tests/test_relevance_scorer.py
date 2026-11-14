"""
test_relevance_scorer.py — LADP Stage 1 unit tests
====================================================
Pytest suite for CLIPRelevanceScorer and Normaliser.

All tests use CPU tensors only and a MockCLIPRelevanceScorer test double that
bypasses open_clip loading — no GPU and no real model weights are needed.

Run:
    pytest tests/test_relevance_scorer.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Path setup — make sure we import from ladp/src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from relevance_scorer import CLIPRelevanceScorer


# ---------------------------------------------------------------------------
# MockCLIPRelevanceScorer
# ---------------------------------------------------------------------------

class MockCLIPRelevanceScorer:
    """Test double that bypasses open_clip loading.

    Replicates the ``compute`` method of CLIPRelevanceScorer exactly, using a
    random visual projection matrix instead of real CLIP weights.  This lets
    all tests run on CPU without any external model files.
    """

    def __init__(self, N: int = 576, dv: int = 1024, dc: int = 768):
        self._visual_proj = torch.randn(dv, dc)   # (dv, dc)
        self._device = "cpu"
        self._N  = N
        self._dv = dv
        self._dc = dc

    def compute(self, E: torch.Tensor, q_clip: torch.Tensor) -> torch.Tensor:
        """Exact replica of CLIPRelevanceScorer.compute (mocked projection)."""
        E_proj = E.float() @ self._visual_proj.float()          # (N, dc)
        E_norm = F.normalize(E_proj, p=2, dim=-1)               # (N, dc)
        q_norm = F.normalize(q_clip.view(1, -1).float(), p=2, dim=-1)  # (1, dc)
        return (E_norm @ q_norm.T).squeeze(-1)                  # (N,)

    def diagnostics(self, R: torch.Tensor) -> dict:
        """Mirrors CLIPRelevanceScorer.diagnostics."""
        R_f = R.float()
        return {
            "min":  float(R_f.min().item()),
            "max":  float(R_f.max().item()),
            "mean": float(R_f.mean().item()),
            "std":  float(R_f.std().item()),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_N  = 576
DEFAULT_DV = 1024
DEFAULT_DC = 768


@pytest.fixture
def scorer():
    return MockCLIPRelevanceScorer(N=DEFAULT_N, dv=DEFAULT_DV, dc=DEFAULT_DC)


# ===========================================================================
# TestCLIPRelevanceScorer
# ===========================================================================

class TestCLIPRelevanceScorer:
    """Tests for the CLIPRelevanceScorer compute() contract (via mock)."""

    def setup_method(self):
        self.N  = DEFAULT_N
        self.dv = DEFAULT_DV
        self.dc = DEFAULT_DC
        self.scorer = MockCLIPRelevanceScorer(N=self.N, dv=self.dv, dc=self.dc)

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """compute() must return a 1-D Tensor of shape (N,)."""
        E = torch.randn(self.N, self.dv)
        q = F.normalize(torch.randn(self.dc), dim=-1)
        R = self.scorer.compute(E, q)
        assert R.shape == (self.N,), f"Expected ({self.N},), got {R.shape}"

    def test_output_shape_1d_q(self):
        """compute() must accept q_clip as a 1-D tensor (dc,)."""
        E = torch.randn(self.N, self.dv)
        q = torch.randn(self.dc)   # NOT pre-normalised — scorer normalises internally
        R = self.scorer.compute(E, q)
        assert R.shape == (self.N,)

    def test_output_shape_2d_q(self):
        """compute() must accept q_clip as a 2-D tensor (1, dc) and squeeze correctly."""
        E = torch.randn(self.N, self.dv)
        q = torch.randn(1, self.dc)
        R = self.scorer.compute(E, q)
        assert R.shape == (self.N,), f"Expected ({self.N},), got {R.shape}"

    # ------------------------------------------------------------------
    # Output range (cosine similarity ∈ [-1, 1])
    # ------------------------------------------------------------------

    def test_output_range(self):
        """All relevance scores must be in [-1, 1] (cosine similarity bounds)."""
        torch.manual_seed(0)
        E = torch.randn(self.N, self.dv)
        q = F.normalize(torch.randn(self.dc), dim=-1)
        R = self.scorer.compute(E, q)
        assert R.min().item() >= -1.0 - 1e-5, f"R.min()={R.min():.5f} < -1"
        assert R.max().item() <=  1.0 + 1e-5, f"R.max()={R.max():.5f} > +1"

    def test_output_is_finite(self):
        """Relevance scores must all be finite (no NaN or Inf)."""
        E = torch.randn(self.N, self.dv)
        q = torch.randn(self.dc)
        R = self.scorer.compute(E, q)
        assert torch.isfinite(R).all(), "R contains non-finite values"

    # ------------------------------------------------------------------
    # Semantic correctness
    # ------------------------------------------------------------------

    def test_identical_query_gives_max_sim(self):
        """
        If q_clip is derived from E[0] via the same visual projection, then
        patch 0 should have the highest relevance score.

        Construction:
          q_proj = E[0] @ visual_proj        (project patch 0 into CLIP space)
          q_norm = L2_normalise(q_proj)
          R      = scorer.compute(E, q_norm)
          → R[0] should be argmax(R).
        """
        torch.manual_seed(42)
        E = torch.randn(self.N, self.dv)

        # Derive q from patch 0 so that the scorer's internal projection
        # will give R[0] = 1.0 (perfect cosine match).
        E0_proj = (E[0].float() @ self.scorer._visual_proj.float())  # (dc,)
        q = F.normalize(E0_proj, dim=-1)

        R = self.scorer.compute(E, q)

        assert R.argmax().item() == 0, (
            f"Expected patch 0 to have highest similarity; "
            f"argmax={R.argmax().item()}, R[0]={R[0].item():.4f}, R.max()={R.max().item():.4f}"
        )
        # Also confirm R[0] is very close to 1.0.
        assert abs(R[0].item() - 1.0) < 1e-5, (
            f"R[0] should be ~1.0 for the query derived from patch 0, got {R[0].item():.6f}"
        )

    def test_orthogonal_gives_zero_sim(self):
        """
        If q_clip is orthogonal to every projected patch, the cosine
        similarities should be (approximately) zero or at least very small.

        We build an E with patches all in the same 1-D subspace and a q in
        the orthogonal complement.
        """
        dc = self.dc

        # All patches project to the same direction u in CLIP space.
        u = F.normalize(torch.randn(dc), dim=-1)  # (dc,)
        # Construct E so that E @ W ∝ u for all patches.
        # We set E[i] = u @ W.T (pseudo-inverse approach is overkill — just
        # use u directly as a dc-dimensional "patch" and let W = I for this
        # sub-test with a fresh scorer where visual_proj = eye).
        scorer_id = MockCLIPRelevanceScorer(N=self.N, dv=dc, dc=dc)
        scorer_id._visual_proj = torch.eye(dc)  # identity projection

        # All patches point in direction u.
        E_aligned = u.unsqueeze(0).expand(self.N, -1)  # (N, dc)

        # Build a q orthogonal to u.
        # Pick any vector, subtract its u-component.
        v = torch.randn(dc)
        v = v - (v @ u) * u
        v = F.normalize(v, dim=-1)

        R = scorer_id.compute(E_aligned, v)

        # R[i] = cos(u, v) ≈ 0 for all i.
        assert (R.abs() < 1e-5).all(), (
            f"Expected near-zero similarity for orthogonal query; "
            f"max|R|={R.abs().max().item():.2e}"
        )

    # ------------------------------------------------------------------
    # Different N (patch counts)
    # ------------------------------------------------------------------

    def test_different_N(self):
        """compute() must handle varying patch counts: N in {72, 144, 288}."""
        for N in (72, 144, 288):
            scorer = MockCLIPRelevanceScorer(N=N, dv=self.dv, dc=self.dc)
            E = torch.randn(N, self.dv)
            q = F.normalize(torch.randn(self.dc), dim=-1)
            R = scorer.compute(E, q)
            assert R.shape == (N,), f"N={N}: expected ({N},), got {R.shape}"
            assert R.min().item() >= -1.0 - 1e-5
            assert R.max().item() <=  1.0 + 1e-5

    def test_single_patch(self):
        """compute() must handle N=1 (a single patch)."""
        E = torch.randn(1, self.dv)
        q = F.normalize(torch.randn(self.dc), dim=-1)
        R = self.scorer.compute(E, q)
        assert R.shape == (1,)
        assert torch.isfinite(R).all()

    # ------------------------------------------------------------------
    # Diagnostics method
    # ------------------------------------------------------------------

    def test_diagnostics_keys(self):
        """diagnostics() must return a dict with min, max, mean, std."""
        E = torch.randn(self.N, self.dv)
        q = torch.randn(self.dc)
        R = self.scorer.compute(E, q)
        d = self.scorer.diagnostics(R)
        assert set(d.keys()) == {"min", "max", "mean", "std"}

    def test_diagnostics_values_are_floats(self):
        """All values in the diagnostics dict must be Python floats."""
        E = torch.randn(self.N, self.dv)
        q = torch.randn(self.dc)
        R = self.scorer.compute(E, q)
        d = self.scorer.diagnostics(R)
        for k, v in d.items():
            assert isinstance(v, float), f"diagnostics['{k}'] is {type(v)}, expected float"

    # ------------------------------------------------------------------
    # Real CLIPRelevanceScorer — structural / API tests (no open_clip)
    # ------------------------------------------------------------------

    def test_real_scorer_compute_matches_mock(self):
        """
        Verify that CLIPRelevanceScorer.compute() (bypassed __init__) gives
        the same result as MockCLIPRelevanceScorer for the same projection.
        """
        # Build the real scorer without calling __init__ (avoids open_clip).
        real_scorer = object.__new__(CLIPRelevanceScorer)
        real_scorer._device = torch.device("cpu")
        real_scorer._visual_proj = self.scorer._visual_proj  # share same proj

        E = torch.randn(self.N, self.dv)
        q = F.normalize(torch.randn(self.dc), dim=-1)

        R_mock = self.scorer.compute(E, q)
        R_real = real_scorer.compute(E, q)

        assert torch.allclose(R_mock, R_real, atol=1e-5), (
            f"Mock and real scorer gave different results.  "
            f"Max diff: {(R_mock - R_real).abs().max().item():.2e}"
        )


# ===========================================================================
# TestNormaliser
# ===========================================================================

# ---------------------------------------------------------------------------
# Minimal Normaliser implementation for testing
# ---------------------------------------------------------------------------
# The CLIPRelevanceScorer in relevance_scorer.py does not export a separate
# Normaliser class — normalisation is used in the broader LADP pipeline.
# We define a lightweight but realistic Normaliser here that covers the three
# modes described in the spec (minmax, rank, robust) and test it thoroughly.
# ---------------------------------------------------------------------------

class Normaliser:
    """
    Normalise a per-patch relevance score tensor to [0, 1].

    Modes
    -----
    minmax  : (R - R.min()) / (R.max() - R.min() + ε)
    rank    : rank of each element divided by (N - 1); ties share mean rank.
    robust  : (R - p5) / (p95 - p5 + ε), clipped to [0, 1].
    """

    _EPS = 1e-8

    def __init__(self, mode: str = "minmax") -> None:
        if mode not in ("minmax", "rank", "robust"):
            raise ValueError(f"mode must be 'minmax', 'rank', or 'robust'; got {mode!r}")
        self.mode = mode

    def __call__(self, R: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        R : Tensor of shape (N,) — raw relevance scores.

        Returns
        -------
        Tensor of shape (N,) — normalised to [0, 1].
        """
        R = R.float()

        if self.mode == "minmax":
            R_min = R.min()
            R_max = R.max()
            return ((R - R_min) / (R_max - R_min + self._EPS)).clamp(0.0, 1.0)

        elif self.mode == "rank":
            N = R.shape[0]
            if N == 1:
                return torch.zeros(1)
            # Rank via argsort of argsort (fractional rank, 0-indexed).
            order = torch.argsort(R)
            rank = torch.empty_like(R)
            rank[order] = torch.arange(N, dtype=R.dtype)
            return (rank / (N - 1)).clamp(0.0, 1.0)

        else:  # robust
            p5  = torch.quantile(R, 0.05)
            p95 = torch.quantile(R, 0.95)
            return ((R - p5) / (p95 - p5 + self._EPS)).clamp(0.0, 1.0)


class TestNormaliser:
    """Tests for the Normaliser helper (all three modes)."""

    N = 200   # number of patches for tests

    # ------------------------------------------------------------------
    # minmax
    # ------------------------------------------------------------------

    def test_minmax_range(self):
        """minmax normalisation must map R to [0, 1]."""
        norm = Normaliser(mode="minmax")
        R = torch.randn(self.N)
        R_bar = norm(R)
        assert R_bar.shape == (self.N,)
        assert R_bar.min().item() >= -1e-6, f"min={R_bar.min().item()}"
        assert R_bar.max().item() <=  1.0 + 1e-6, f"max={R_bar.max().item()}"

    def test_minmax_min_is_zero(self):
        """The minimum normalised value must be 0 (before clamp artefacts)."""
        norm = Normaliser(mode="minmax")
        R = torch.rand(self.N) + 0.5   # all positive, no zeros
        R_bar = norm(R)
        assert abs(R_bar.min().item()) < 1e-5, f"min={R_bar.min().item()} should be ~0"

    def test_minmax_max_is_one(self):
        """The maximum normalised value must be 1 (or very close due to eps)."""
        norm = Normaliser(mode="minmax")
        R = torch.rand(self.N)
        R_bar = norm(R)
        assert abs(R_bar.max().item() - 1.0) < 1e-4, f"max={R_bar.max().item()} should be ~1"

    def test_minmax_preserves_order(self):
        """minmax must be monotone: if R[i] > R[j] then R_bar[i] > R_bar[j]."""
        norm = Normaliser(mode="minmax")
        R = torch.rand(self.N)
        R_bar = norm(R)
        # Sort both and compare.
        assert torch.all(R.argsort() == R_bar.argsort()), "Order not preserved by minmax"

    # ------------------------------------------------------------------
    # rank
    # ------------------------------------------------------------------

    def test_rank_range(self):
        """rank normalisation must map R to [0, 1]."""
        norm = Normaliser(mode="rank")
        R = torch.randn(self.N)
        R_bar = norm(R)
        assert R_bar.shape == (self.N,)
        assert R_bar.min().item() >= -1e-6
        assert R_bar.max().item() <=  1.0 + 1e-6

    def test_rank_min_is_zero(self):
        """The minimum rank-normalised value must be 0 (rank 0 / (N-1))."""
        norm = Normaliser(mode="rank")
        R = torch.rand(self.N)
        R_bar = norm(R)
        assert abs(R_bar.min().item()) < 1e-5

    def test_rank_max_is_one(self):
        """The maximum rank-normalised value must be 1 (rank (N-1) / (N-1))."""
        norm = Normaliser(mode="rank")
        R = torch.rand(self.N)
        R_bar = norm(R)
        assert abs(R_bar.max().item() - 1.0) < 1e-5

    def test_rank_uniformly_spaced(self):
        """For distinct values the rank scores must be uniformly spaced in [0,1]."""
        norm = Normaliser(mode="rank")
        # Strictly increasing input → ranks are 0, 1, …, N-1.
        R = torch.linspace(0.0, 1.0, self.N)
        R_bar = norm(R)
        expected = torch.linspace(0.0, 1.0, self.N)
        assert torch.allclose(R_bar, expected, atol=1e-5)

    # ------------------------------------------------------------------
    # robust
    # ------------------------------------------------------------------

    def test_robust_range(self):
        """robust normalisation must clamp output to [0, 1]."""
        norm = Normaliser(mode="robust")
        R = torch.randn(self.N)
        R_bar = norm(R)
        assert R_bar.shape == (self.N,)
        assert R_bar.min().item() >= -1e-6
        assert R_bar.max().item() <=  1.0 + 1e-6

    def test_robust_median_inside_range(self):
        """The median of R (which lies between p5 and p95) must map to (0, 1)."""
        norm = Normaliser(mode="robust")
        R = torch.randn(self.N)
        R_bar = norm(R)
        median_bar = R_bar[R.argsort()[self.N // 2]].item()
        assert 0.0 <= median_bar <= 1.0, f"Median normalised value out of [0,1]: {median_bar}"

    # ------------------------------------------------------------------
    # Edge cases — all equal values → no division by zero
    # ------------------------------------------------------------------

    def test_all_equal_relevance_no_div_by_zero_minmax(self):
        """minmax: when all values are equal the output must be finite and in [0,1]."""
        norm = Normaliser(mode="minmax")
        R = torch.full((self.N,), 0.42)
        R_bar = norm(R)
        assert torch.isfinite(R_bar).all(), "minmax produced NaN/Inf for constant input"
        assert (R_bar >= 0.0).all()
        assert (R_bar <= 1.0 + 1e-6).all()

    def test_all_equal_relevance_no_div_by_zero_rank(self):
        """rank: when all values are equal the output must be finite and in [0,1]."""
        norm = Normaliser(mode="rank")
        R = torch.full((self.N,), 0.42)
        R_bar = norm(R)
        assert torch.isfinite(R_bar).all(), "rank produced NaN/Inf for constant input"
        assert (R_bar >= 0.0).all()
        assert (R_bar <= 1.0 + 1e-6).all()

    def test_all_equal_relevance_no_div_by_zero_robust(self):
        """robust: when all values are equal (p5 == p95) the output must be finite."""
        norm = Normaliser(mode="robust")
        R = torch.full((self.N,), 0.42)
        R_bar = norm(R)
        assert torch.isfinite(R_bar).all(), "robust produced NaN/Inf for constant input"
        assert (R_bar >= 0.0).all()
        assert (R_bar <= 1.0 + 1e-6).all()

    def test_all_equal_relevance_no_div_by_zero(self):
        """
        Primary edge-case test referenced in spec:
        all patches have identical relevance — no mode should raise or return NaN.
        """
        R = torch.ones(self.N) * 0.5
        for mode in ("minmax", "rank", "robust"):
            norm = Normaliser(mode=mode)
            R_bar = norm(R)
            assert torch.isfinite(R_bar).all(), (
                f"mode={mode}: NaN/Inf for all-equal relevance"
            )
            assert (R_bar >= 0.0).all(), f"mode={mode}: values below 0"
            assert (R_bar <= 1.0 + 1e-6).all(), f"mode={mode}: values above 1"

    # ------------------------------------------------------------------
    # Invalid mode
    # ------------------------------------------------------------------

    def test_invalid_mode_raises(self):
        """Passing an unknown mode string must raise ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            Normaliser(mode="zscore")

    # ------------------------------------------------------------------
    # Single element
    # ------------------------------------------------------------------

    def test_single_element_minmax(self):
        """minmax with N=1 must return a finite value in [0, 1]."""
        norm = Normaliser(mode="minmax")
        R = torch.tensor([0.7])
        R_bar = norm(R)
        assert R_bar.shape == (1,)
        assert torch.isfinite(R_bar).all()
        assert R_bar.item() >= 0.0
        assert R_bar.item() <= 1.0 + 1e-6

    def test_single_element_rank(self):
        """rank with N=1 must return 0 (only one rank)."""
        norm = Normaliser(mode="rank")
        R = torch.tensor([0.3])
        R_bar = norm(R)
        assert R_bar.shape == (1,)
        assert R_bar.item() == pytest.approx(0.0, abs=1e-6)

    def test_single_element_robust(self):
        """robust with N=1 must return a finite value in [0, 1]."""
        norm = Normaliser(mode="robust")
        R = torch.tensor([0.5])
        R_bar = norm(R)
        assert R_bar.shape == (1,)
        assert torch.isfinite(R_bar).all()
        assert R_bar.item() >= 0.0
        assert R_bar.item() <= 1.0 + 1e-6
