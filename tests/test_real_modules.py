"""
Audit M6: tests of the real modules (no copied implementations):
src/normaliser.py, src/priority_module.py, src/diagnostics/knor.py.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.loss_estimator import NormStats  # noqa: E402
from src.normaliser import Normaliser, validate_scores  # noqa: E402
from src.priority_module import PriorityModule  # noqa: E402
from diagnostics.knor import KNORDiagnostic, correlate_loss_with_error, correlate_loss_with_vqa  # noqa: E402


def _ns(p5=0.1, p95=0.5):
    return NormStats(torch.zeros(2), torch.ones(2), torch.zeros(2), torch.ones(2), p5, p95)


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class TestNormaliser:
    nm = Normaliser()

    def test_robust_uses_train_percentiles_and_clips(self):
        L = torch.tensor([0.0, 0.1, 0.3, 0.5, 2.0])
        assert torch.allclose(self.nm.normalise_loss(L, _ns()), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]), atol=1e-6)

    def test_robust_does_not_inflate_easy_images(self):
        # all patches well reconstructed -> all near 0 (per-image min-max would spread them to [0,1])
        L = torch.tensor([0.01, 0.02, 0.03])
        assert self.nm.normalise_loss(L, _ns()).max() == 0.0
        assert self.nm.normalise_loss(L, _ns(), mode="per_image").max() == pytest.approx(1.0)

    def test_per_image_constant_input_is_finite(self):
        out = self.nm.normalise_loss(torch.full((5,), 0.7), _ns(), mode="per_image")
        assert torch.isfinite(out).all() and out.abs().max() == 0.0

    def test_relevance_minmax_and_rank(self):
        R = torch.tensor([0.2, -0.1, 0.5, 0.3])
        mm = self.nm.normalise_relevance(R, "minmax")
        assert mm.min() == 0.0 and mm.max() == pytest.approx(1.0) and int(mm.argmax()) == 2
        rk = self.nm.normalise_relevance(R, "rank")
        assert torch.allclose(rk, torch.tensor([1, 0, 3, 2]) / 3.0)

    def test_relevance_single_patch_and_constant(self):
        assert self.nm.normalise_relevance(torch.tensor([0.4]), "rank").tolist() == [0.0]
        assert torch.isfinite(self.nm.normalise_relevance(torch.full((4,), 0.3), "minmax")).all()

    @pytest.mark.parametrize("fn,arg", [("normalise_loss", "softmax"), ("normalise_relevance", "robust")])
    def test_invalid_modes_raise(self, fn, arg):
        with pytest.raises(ValueError):
            if fn == "normalise_loss":
                self.nm.normalise_loss(torch.rand(3), _ns(), mode=arg)
            else:
                self.nm.normalise_relevance(torch.rand(3), mode=arg)

    def test_validate_scores(self):
        assert validate_scores(torch.tensor([0.0, 1.0]), "x")
        with pytest.raises(RuntimeError, match="outside"):
            validate_scores(torch.tensor([1.5]), "x")
        with pytest.raises(RuntimeError, match="NaN"):
            validate_scores(torch.tensor([float("nan")]), "x")


# ---------------------------------------------------------------------------
# PriorityModule
# ---------------------------------------------------------------------------

class TestPriority:
    L = torch.tensor([0.9, 0.1, 0.5, 0.0])
    R = torch.tensor([0.1, 0.9, 0.5, 1.0])

    def test_additive_and_extremes(self):
        assert torch.allclose(PriorityModule("additive", 0.5, 0.5).forward(self.L, self.R), (self.L + self.R) / 2)
        assert torch.equal(PriorityModule("additive", 1.0, 0.0).forward(self.L, self.R), self.L)
        assert torch.equal(PriorityModule("additive", 0.0, 1.0).forward(self.L, self.R), self.R)

    def test_multiplicative(self):
        assert torch.allclose(PriorityModule("multiplicative").forward(self.L, self.R), self.L * self.R)

    @pytest.mark.parametrize("kw", [dict(mode="max"), dict(alpha=-0.1, beta=1.1), dict(alpha=0.6, beta=0.6)])
    def test_invalid_configuration(self, kw):
        with pytest.raises(ValueError):
            PriorityModule(**kw)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            PriorityModule().forward(torch.rand(3), torch.rand(4))

    def test_top_k_keeps_highest_in_original_order(self):
        S = torch.tensor([0.2, 0.9, 0.1, 0.8, 0.5])
        pm = PriorityModule()
        assert pm.top_k_indices(S, 3).tolist() == [1, 3, 4]
        assert pm.top_k_indices(S, 5).tolist() == [0, 1, 2, 3, 4]
        assert pm.rank(S).tolist() == [1, 3, 4, 0, 2]

    @pytest.mark.parametrize("k", [0, 6])
    def test_top_k_bounds(self, k):
        with pytest.raises(ValueError):
            PriorityModule().top_k_indices(torch.rand(5), k)


# ---------------------------------------------------------------------------
# KNOR and correlations
# ---------------------------------------------------------------------------

class TestKNOR:
    def test_identical_spaces_give_one(self):
        X = torch.randn(30, 8)
        assert KNORDiagnostic(5).compute(X, X.clone())["knor_mean"] == pytest.approx(1.0)

    def test_scaling_invariance(self):
        X = torch.randn(30, 8)
        assert KNORDiagnostic(5).compute(X, 7.0 * X)["knor_mean"] == pytest.approx(1.0)

    def test_independent_spaces_near_chance(self):
        torch.manual_seed(0)
        r = KNORDiagnostic(10).compute(torch.randn(400, 32), torch.randn(400, 32))
        assert 0.0 <= r["knor_mean"] < 0.15                           # chance = k/(M-1) ~ 0.025

    def test_self_never_counted_as_neighbour_with_duplicates(self):
        torch.manual_seed(1)
        X = torch.randn(20, 6)
        X[7] = X[2]                                                   # duplicate image
        Y = torch.randn(20, 6)
        Y[7] = Y[2]
        d = KNORDiagnostic(3)
        sim = torch.nn.functional.normalize(X, dim=-1) @ torch.nn.functional.normalize(X, dim=-1).T
        sim.fill_diagonal_(float("-inf"))
        assert 7 in sim[2].topk(3).indices.tolist()                   # duplicate is a true neighbour
        r = d.compute(X, Y)
        assert 0.0 <= float(r["knor_per_image"][2]) <= 1.0
        assert float(r["knor_per_image"][2]) >= 1 / 3                 # the shared duplicate overlaps

    @pytest.mark.parametrize("E,T,k", [(torch.randn(5, 3), torch.randn(5, 3), 5),
                                       (torch.randn(5, 3), torch.randn(6, 3), 2)])
    def test_invalid_inputs(self, E, T, k):
        with pytest.raises(ValueError):
            KNORDiagnostic(k).compute(E, T)

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            KNORDiagnostic(0)


class TestCorrelations:
    def test_sign_higher_loss_more_errors_is_positive(self):
        losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        correct = [True, True, True, False, False, False]
        r = correlate_loss_with_error(losses, correct)
        assert r["pearson_r"] > 0.8 and r["mean_loss_wrong"] > r["mean_loss_correct"] and not r["degenerate"]

    def test_constant_outcome_flags_degenerate(self):
        r = correlate_loss_with_error([0.1, 0.2, 0.3, 0.4], [True] * 4)
        assert r["degenerate"] is True and r["pearson_r"] != r["pearson_r"]   # NaN, flagged

    def test_vqa_version_uses_graded_error(self):
        r = correlate_loss_with_vqa([0.1, 0.2, 0.3, 0.4], [1.0, 0.9, 0.3, 0.0])
        assert r["pearson_r"] > 0.9 and r["outcome"].startswith("vqa_error")

    @pytest.mark.parametrize("losses,accs", [([0.1, 0.2], [1.0, 0.0]), ([0.1, 0.2, 0.3], [1.0, 0.0]),
                                             ([0.1, 0.2, 0.3], [1.0, 2.0, 0.0])])
    def test_vqa_version_validates(self, losses, accs):
        with pytest.raises(ValueError):
            correlate_loss_with_vqa(losses, accs)
