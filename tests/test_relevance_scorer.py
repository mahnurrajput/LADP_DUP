"""
test_relevance_scorer.py — LADP
===============================
Tests of the real ``CLIPRelevanceScorer`` (audit H3) and the ablation
``LLMEmbeddingRelevanceScorer``.  CPU only, no downloaded weights: the scorer is
built from components (``from_components``), and the projection pipeline is
checked against open_clip's own ``VisionTransformer`` with random weights.

The normaliser is tested on the real module in tests/test_real_modules.py
(this file used to test a local copy of it).
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from relevance_scorer import CLIPRelevanceScorer, LLMEmbeddingRelevanceScorer  # noqa: E402

W, D = 32, 16          # tiny stand-ins for width 1024 / joint dim 768


def _scorer(seed=0, width=W, dim=D, encode_text=None):
    torch.manual_seed(seed)
    ln = torch.nn.LayerNorm(width)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.normal_(0, 0.1)
    return CLIPRelevanceScorer.from_components(ln, torch.randn(width, dim), encode_text=encode_text)


class TestProjectionPipeline:
    def test_matches_manual_formula(self):
        s = _scorer()
        X, q = torch.randn(10, W), torch.randn(D)
        expected = F.normalize(s._ln_post(X) @ s._visual_proj, dim=-1) @ F.normalize(q, dim=0)
        assert torch.allclose(s.compute(X, q), expected, atol=1e-6)

    def test_ln_post_is_applied(self):
        s = _scorer()
        X, q = torch.randn(10, W), torch.randn(D)
        without_ln = F.normalize(X @ s._visual_proj, dim=-1) @ F.normalize(q, dim=0)
        assert not torch.allclose(s.compute(X, q), without_ln, atol=1e-3)

    def test_against_open_clip_vision_transformer(self):
        open_clip = pytest.importorskip("open_clip")
        from open_clip.transformer import VisionTransformer
        torch.manual_seed(0)
        vt = VisionTransformer(image_size=28, patch_size=14, width=W, layers=2, heads=4,
                               mlp_ratio=2.0, output_dim=D, output_tokens=True).eval()
        assert vt.pool_type == "tok" and not vt.final_ln_after_pool and vt.attn_pool is None
        s = CLIPRelevanceScorer.from_components(vt.ln_post, vt.proj)
        x = torch.randn(1, 3, 28, 28)
        with torch.no_grad():
            pooled, tokens = vt(x)                       # CLIP's own outputs
            pre = vt.transformer(vt._embeds(x))[0]       # final-layer tokens before ln_post
        # applied to CLS, the formula reproduces CLIP's image embedding exactly
        assert torch.allclose(s.patch_embeddings(pre[:1]), F.normalize(pooled, dim=-1), atol=1e-5)
        # applied to the patches, it equals CLIP's ln_post'ed patch tokens through proj
        assert torch.allclose(s.patch_embeddings(pre[1:]), F.normalize(tokens[0] @ vt.proj, dim=-1), atol=1e-5)

    def test_patch_matching_the_question_scores_highest(self):
        s = _scorer()
        X = torch.randn(12, W)
        q = s.patch_embeddings(X)[5]
        R = s.compute(X, q)
        assert int(R.argmax()) == 5 and R[5] == pytest.approx(1.0, abs=1e-5)

    def test_range_shape_and_finiteness(self):
        s = _scorer()
        for n in (1, 7, 576):
            R = s.compute(torch.randn(n, W), torch.randn(1, D))
            assert R.shape == (n,) and torch.isfinite(R).all()
            assert R.min() >= -1 - 1e-5 and R.max() <= 1 + 1e-5

    def test_fp16_tokens_accepted(self):
        s = _scorer()
        X, q = torch.randn(4, W), torch.randn(D)
        assert torch.allclose(s.compute(X.half(), q), s.compute(X, q), atol=2e-3)


class TestValidation:
    @pytest.mark.parametrize("X", [torch.randn(4, W + 1), torch.randn(W), torch.randn(2, 4, W)])
    def test_wrong_token_shape(self, X):
        with pytest.raises(ValueError, match="final-layer tokens"):
            _scorer().compute(X, torch.randn(D))

    def test_wrong_question_dim(self):
        with pytest.raises(ValueError, match="question embedding"):
            _scorer().compute(torch.randn(3, W), torch.randn(D + 1))

    def test_ln_post_width_must_match_proj(self):
        with pytest.raises(ValueError, match="ln_post"):
            CLIPRelevanceScorer.from_components(torch.nn.LayerNorm(W + 1), torch.randn(W, D))

    def test_proj_must_be_2d(self):
        with pytest.raises(ValueError):
            CLIPRelevanceScorer.from_components(torch.nn.LayerNorm(W), torch.randn(W))

    def test_encode_question_needs_a_text_encoder(self):
        with pytest.raises(RuntimeError, match="text encoder"):
            _scorer().encode_question("what is this?")

    def test_encode_question_checks_dim_and_normalises(self):
        s = _scorer(encode_text=lambda q: torch.full((D,), 3.0))
        q = s.encode_question("what?")
        assert q.shape == (D,) and q.norm() == pytest.approx(1.0)
        bad = _scorer(encode_text=lambda q: torch.ones(D + 2))
        with pytest.raises(ValueError, match="text embedding"):
            bad.encode_question("what?")

    def test_compute_batch(self):
        s = _scorer(encode_text=lambda q: torch.randn(D))
        out = s.compute_batch([torch.randn(5, W), torch.randn(3, W)], ["a", "b"])
        assert [r.shape[0] for r in out] == [5, 3]
        with pytest.raises(ValueError):
            s.compute_batch([torch.randn(5, W)], ["a", "b"])

    def test_diagnostics(self):
        d = _scorer().diagnostics(torch.tensor([0.1, 0.5, -0.2]))
        assert set(d) == {"min", "max", "mean", "std"} and all(isinstance(v, float) for v in d.values())


class TestLLMEmbeddingScorer:
    def test_scaled_dot_product(self):
        class Emb(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(50, 8), requires_grad=False)

            def forward(self, ids):
                return self.weight[ids]

        lm = type("LM", (), {})()
        lm.model = type("M", (), {"embed_tokens": Emb()})()
        tok = lambda text, **kw: {"input_ids": torch.tensor([[1, 2, 3]])}  # noqa: E731
        s = LLMEmbeddingRelevanceScorer(lm, tok, "cpu")
        q = s.encode_question("x")
        T = torch.randn(4, 8)
        assert torch.allclose(s.compute(T, q), T @ q / math.sqrt(8))
