"""
relevance_scorer.py — LADP project
====================================
Implements two relevance scorers for the Loss-Aware Dynamic Projection (LADP)
pipeline.

CLIPRelevanceScorer  (primary scorer)
-------------------------------------
Rᵢ = cos( ln_post(Xᵢ) · W_proj ,  q_CLIP )  for every patch i, where

  X        (N, 1024)  FINAL-layer CLIP ViT-L/14-336 patch tokens, CLS removed,
                      *before* the final LayerNorm.  In LLaVA this is the
                      vision tower's ``last_hidden_state[:, 1:]``
                      (``src/feature_capture.py`` records it as ``E_final``).
  ln_post  LayerNorm(1024)   CLIP's final LayerNorm (open_clip ``visual.ln_post``,
                      = HF ``vision_model.post_layernorm``).
  W_proj   (1024, 768)       CLIP's image projection (open_clip ``visual.proj``).
  q_CLIP   (768,)            CLIP text embedding of the question.
  Both sides are L2-normalised, so Rᵢ ∈ [-1, 1].

Why this form (verified against open_clip's VisionTransformer, ViT-L-14-336:
pool_type "tok", final_ln_after_pool False, no attentional pooling):
CLIP's own image embedding is  ln_post(x)[:, 0] @ proj  — ln_post is applied to
*all* final-layer tokens and proj maps width 1024 to the joint 768-d space.
Applying the same two operations to the patch tokens x[:, 1:] puts each patch
in the space the text encoder was trained to match; for the CLS token the
formula reproduces CLIP's image embedding exactly (tested).

The earlier version applied proj to the *penultimate*-layer tokens without
ln_post (audit H3): a different layer and an un-normalised input, so its
scores were not in CLIP's image–text space.

Caveat (research decision, not settled by code): final-layer CLIP patch tokens
are known to localise weakly, because CLIP was only trained on the pooled CLS
token.  Whether this dense form, a value-path variant (MaskCLIP-style) or
another feature works best must be measured against the VizWiz answer-grounding
masks on the tuning split (scripts/validate_relevance_grounding.py).

  By contrast, the LLM-space approach (post-connector T in R^4096 vs.
  mean-token LLM embeddings) suffers from a modality gap: the LLM was not
  trained to align image tokens with text in a cosine-metric sense.

LLMEmbeddingRelevanceScorer  (ABLATION ONLY — unfixed baseline)
----------------------------------------------------------------
Rᵢ = (Tᵢ · q_llm) / sqrt(d)  where T is post-connector (R^4096) and
q_llm is the mean LLM input embedding of question tokens.  Included so
that the ablation study can compare CLIP-based vs. LLM-space relevance.

Dependencies:
  open_clip_torch  — pip install open-clip-torch
  torch            — pip install torch
  transformers     — pip install transformers  (for LLMEmbeddingRelevanceScorer)
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIPRelevanceScorer — primary (fixed) scorer
# ---------------------------------------------------------------------------


class CLIPRelevanceScorer:
    """Score the relevance of each pre-connector patch to the question using
    the shared CLIP embedding space.

    Parameters
    ----------
    clip_model_name : str
        Open-CLIP model architecture name, e.g. ``"ViT-L-14"``.
    pretrained : str
        Open-CLIP pretrained weights tag, e.g. ``"openai"`` for the original
        OpenAI-released weights (which match the ones used in LLaVA-1.5).
    device : str | torch.device
        Device on which to run CLIP inference, e.g. ``"cuda"`` or ``"cpu"``.

    Notes
    -----
    Only the CLIP text encoder and the visual projection matrix
    (``model.visual.proj``) are needed for relevance scoring.  The full
    vision encoder (``model.visual``) is NOT used here because the
    pre-connector features E are extracted externally by ``LlavaAdapter``
    and passed directly to :meth:`compute`.  However, ``visual.proj`` must
    be loaded alongside the visual encoder so we keep the full model in
    memory and simply never call the visual forward pass.
    """

    # Dimensions for CLIP ViT-L/14 (OpenAI weights).
    _VISION_DIM: int = 1024  # pre-projection CLIP vision hidden dim
    _CLIP_EMBED_DIM: int = 768  # joint CLIP embedding space dim (post-projection)

    def __init__(
        self,
        clip_model_name: str = "ViT-L-14-336",
        pretrained: str = "openai",
        device: str = "cuda",
    ) -> None:
        import open_clip  # deferred import — optional dependency

        self._device = torch.device(device)
        self._clip_model_name = clip_model_name
        self._pretrained = pretrained

        logger.info(
            "Loading open_clip model '%s' (pretrained='%s') on %s …",
            clip_model_name,
            pretrained,
            self._device,
        )

        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=pretrained,
            device=self._device,
        )
        model.eval()

        # Disable gradient tracking for all parameters — this module is
        # always used in inference mode.
        for param in model.parameters():
            param.requires_grad_(False)

        v = model.visual
        # The dense formula ln_post(x) @ proj is only CLIP's own readout for
        # this pooling configuration (checked in open_clip's VisionTransformer).
        if getattr(v, "proj", None) is None or getattr(v, "attn_pool", None) is not None \
                or getattr(v, "pool_type", "tok") != "tok" or getattr(v, "final_ln_after_pool", False):
            raise RuntimeError(
                f"open_clip model '{clip_model_name}' does not use CLS-token pooling with a "
                "linear proj (pool_type='tok', no attentional pooling, ln_post before pooling); "
                "the patch-embedding formula ln_post(x) @ proj does not apply to it.")
        self._ln_post: torch.nn.LayerNorm = v.ln_post        # LayerNorm(1024)
        self._visual_proj: Tensor = v.proj                   # (1024, 768)
        self._model = model                                  # only encode_text() is called
        self._tokenizer = open_clip.get_tokenizer(clip_model_name)
        self._encode_text_fn = None
        self._check_dims()
        logger.info("CLIPRelevanceScorer ready: ln_post %s, proj %s, text dim %d",
                    tuple(self._ln_post.normalized_shape), tuple(self._visual_proj.shape),
                    self._visual_proj.shape[1])

    # ------------------------------------------------------------------
    @classmethod
    def from_components(cls, ln_post: torch.nn.Module, visual_proj: Tensor,
                        encode_text=None, device: str = "cpu") -> "CLIPRelevanceScorer":
        """Build a scorer from its parts (tests, or weights taken from another CLIP copy).

        ``encode_text(question) -> (768,)`` is only needed for :meth:`encode_question`.
        """
        self = object.__new__(cls)
        self._device = torch.device(device)
        self._clip_model_name, self._pretrained = "components", None
        self._ln_post, self._visual_proj = ln_post, visual_proj
        self._model, self._tokenizer, self._encode_text_fn = None, None, encode_text
        self._check_dims()
        return self

    def _check_dims(self) -> None:
        if self._visual_proj.dim() != 2:
            raise ValueError(f"visual_proj must be 2-D, got {tuple(self._visual_proj.shape)}")
        width = self._visual_proj.shape[0]
        ln_shape = tuple(getattr(self._ln_post, "normalized_shape", (width,)))
        if ln_shape != (width,):
            raise ValueError(f"ln_post normalises {ln_shape} but proj expects width {width}")

    @property
    def width(self) -> int:
        """Vision token width the scorer expects (1024 for ViT-L/14)."""
        return int(self._visual_proj.shape[0])

    @property
    def embed_dim(self) -> int:
        """Joint image-text embedding size (768 for ViT-L/14)."""
        return int(self._visual_proj.shape[1])

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_question(self, question: str) -> Tensor:
        """Encode a question string into the joint CLIP embedding space.

        Parameters
        ----------
        question : str
            The VQA question text.

        Returns
        -------
        Tensor of shape (768,)
            L2-normalised CLIP text embedding for the question.
        """
        if self._model is None:
            if self._encode_text_fn is None:
                raise RuntimeError("No text encoder: build the scorer with open_clip or pass encode_text.")
            q_raw = torch.as_tensor(self._encode_text_fn(question)).view(1, -1).float()
        else:
            tokens: Tensor = self._tokenizer([question]).to(self._device)
            q_raw = self._model.encode_text(tokens)               # (1, 768)
        if q_raw.shape[-1] != self.embed_dim:
            raise ValueError(f"text embedding has {q_raw.shape[-1]} dims, image side {self.embed_dim}")
        return F.normalize(q_raw.float(), p=2, dim=-1).squeeze(0)  # (768,)

    # ------------------------------------------------------------------
    # Per-image relevance scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def patch_embeddings(self, X_final: Tensor) -> Tensor:
        """Patch tokens in CLIP's joint space: L2norm( ln_post(X) @ W_proj ).

        X_final : (N, width) final-layer tokens *before* ln_post
                  (LLaVA: vision tower ``last_hidden_state[:, 1:]``).
        returns : (N, embed_dim), unit norm.
        """
        if X_final.dim() != 2 or X_final.shape[-1] != self.width:
            raise ValueError(f"expected final-layer tokens (N, {self.width}), got {tuple(X_final.shape)}")
        P = self._visual_proj
        ln = self._ln_post
        X = X_final.to(P.device)
        w = getattr(ln, "weight", None)
        Z = ln(X.to(w.dtype if w is not None else torch.float32)).float() @ P.float()   # (N, 768)
        return F.normalize(Z, p=2, dim=-1)

    @torch.no_grad()
    def compute(self, X_final: Tensor, q_clip: Tensor) -> Tensor:
        """Per-patch relevance Rᵢ = cos( ln_post(Xᵢ) W_proj , q ) for one image.

        Parameters
        ----------
        X_final : (N, 1024)  FINAL-layer CLIP patch tokens before ln_post, CLS
                  removed (``LlavaFeatureCapture.E_final``).  NOT the
                  penultimate-layer E that the connector and g_phi use.
        q_clip  : (768,) or (1, 768) text embedding (:meth:`encode_question`).

        Returns
        -------
        (N,) cosine similarities in [-1, 1].

        Steps and shapes
        ----------------
        1. Z  = ln_post(X_final)              (N, 1024)
        2. P  = Z @ W_proj                    (N, 768)
        3. P̂ = P / ||P||,  q̂ = q / ||q||     (N, 768), (1, 768)
        4. R  = P̂ q̂ᵀ                         (N,)
        """
        q = q_clip.reshape(1, -1).float().to(self._visual_proj.device)
        if q.shape[-1] != self.embed_dim:
            raise ValueError(f"question embedding has {q.shape[-1]} dims; expected {self.embed_dim}")
        E_norm = self.patch_embeddings(X_final)                   # (N, 768)
        q_norm = F.normalize(q, p=2, dim=-1)                      # (1, 768)
        R: Tensor = (E_norm @ q_norm.T).squeeze(-1)               # (N,)

        return R

    # ------------------------------------------------------------------
    # Batch convenience wrapper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_batch(
        self,
        E_list: List[Tensor],
        questions: List[str],
    ) -> List[Tensor]:
        """Compute relevance scores for a batch of (image features, question) pairs.

        Parameters
        ----------
        E_list : list of Tensor
            Each element is a Tensor of shape (N, 1024) — FINAL-layer CLIP
            patch tokens before ln_post (``E_final``) for one image.
        questions : list of str
            Question strings, one per image.  Must have the same length as
            ``E_list``.

        Returns
        -------
        list of Tensor
            Each element is a Tensor of shape (N,) — the relevance scores
            for the corresponding image.

        Raises
        ------
        ValueError
            If ``E_list`` and ``questions`` have different lengths.
        """
        if len(E_list) != len(questions):
            raise ValueError(
                f"E_list and questions must have the same length.  "
                f"Got len(E_list)={len(E_list)} and len(questions)={len(questions)}."
            )

        results: List[Tensor] = []
        for E, question in zip(E_list, questions):
            q_clip = self.encode_question(question)
            R = self.compute(E, q_clip)
            results.append(R)

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self, R: Tensor) -> dict:
        """Return basic descriptive statistics for a relevance score vector.

        Parameters
        ----------
        R : Tensor, shape (N,)
            Per-patch relevance scores from :meth:`compute`.

        Returns
        -------
        dict with keys:
            ``"min"``  — float, minimum score.
            ``"max"``  — float, maximum score.
            ``"mean"`` — float, mean score.
            ``"std"``  — float, standard deviation.
        """
        R_f = R.float()
        return {
            "min": float(R_f.min().item()),
            "max": float(R_f.max().item()),
            "mean": float(R_f.mean().item()),
            "std": float(R_f.std().item()),
        }


# ---------------------------------------------------------------------------
# LLMEmbeddingRelevanceScorer — ablation baseline (unfixed approach)
# ---------------------------------------------------------------------------


class LLMEmbeddingRelevanceScorer:
    """ABLATION ONLY — the original (unfixed) approach to relevance scoring.

    Computes Rᵢ = (Tᵢ · q_llm) / sqrt(d) where:
      - T  is the post-connector token matrix, shape (N, dt=4096)
      - q_llm is the mean of LLM input embeddings of question tokens, shape (dt,)

    This approach operates entirely in LLM hidden space and does NOT use the
    joint CLIP embedding space.  It is included as an ablation baseline to
    quantify how much the CLIP-space scorer (CLIPRelevanceScorer) improves
    over the naive LLM-space alternative.

    Known limitations compared to CLIPRelevanceScorer:
      1. T lies in a different distribution than question token embeddings —
         the LLM was not trained to align visual tokens with text in a
         cosine-metric sense (modality gap).
      2. Scaling by 1/sqrt(d) is a heuristic (attention-style), not a
         calibrated similarity.
      3. The question is encoded as mean token embeddings, discarding
         positional and contextual information.

    Parameters
    ----------
    language_model : transformers.PreTrainedModel
        A causal LM (e.g. ``LlamaForCausalLM``) whose
        ``model.embed_tokens`` layer is used to encode the question.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer for ``language_model``.
    device : str | torch.device
        Target device for inference.
    """

    def __init__(self, language_model, tokenizer, device: str | torch.device) -> None:
        self._device = torch.device(device)
        self._language_model = language_model
        self._tokenizer = tokenizer

        # Resolve the token embedding table.  For LlamaForCausalLM (used in
        # LLaVA-1.5-7B) this is at model.model.embed_tokens.
        if hasattr(language_model, "model") and hasattr(
            language_model.model, "embed_tokens"
        ):
            self._embed_tokens = language_model.model.embed_tokens
        elif hasattr(language_model, "embed_tokens"):
            self._embed_tokens = language_model.embed_tokens
        else:
            raise RuntimeError(
                "Cannot locate an embed_tokens layer in the provided "
                "language_model.  Expected either model.embed_tokens or "
                "model.model.embed_tokens (LlamaForCausalLM layout)."
            )

        # Determine the hidden dimension from the embedding weight shape.
        self._dt: int = self._embed_tokens.weight.shape[-1]  # typically 4096

        logger.info(
            "LLMEmbeddingRelevanceScorer (ablation) ready.  dt=%d  device=%s",
            self._dt,
            self._device,
        )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_question(self, question: str) -> Tensor:
        """Encode a question as the mean of its LLM input embeddings.

        Parameters
        ----------
        question : str
            The VQA question text.

        Returns
        -------
        Tensor of shape (dt,) = (4096,)
            Mean of LLM token embeddings for the question tokens.  NOT
            normalised (raw embedding space mean).
        """
        encoded = self._tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids: Tensor = encoded["input_ids"].to(self._device)  # (1, L)
        # Embedding lookup → (1, L, dt).
        token_embeds: Tensor = self._embed_tokens(input_ids)       # (1, L, dt)
        # Mean over the token sequence dimension.
        q_llm: Tensor = token_embeds.squeeze(0).mean(dim=0)        # (dt,)
        return q_llm

    # ------------------------------------------------------------------
    # Per-image relevance scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute(self, T: Tensor, q_llm: Tensor) -> Tensor:
        """Compute per-patch relevance scores using LLM-space dot product.

        Parameters
        ----------
        T : Tensor, shape (N, 4096)
            Post-connector tokens for one image (output of the LLaVA MLP
            projector), as returned by ``LlavaAdapter``.
        q_llm : Tensor, shape (4096,)
            Mean LLM embedding of the question, from :meth:`encode_question`.

        Returns
        -------
        Tensor of shape (N,)
            Scaled dot-product relevance scores Rᵢ = (Tᵢ · q_llm) / sqrt(d).
        """
        T = T.to(self._device)
        q_llm = q_llm.to(self._device)

        # (N, dt) @ (dt,) → (N,), scaled by 1/sqrt(d).
        R: Tensor = (T.float() @ q_llm.float()) / math.sqrt(self._dt)
        return R


# ---------------------------------------------------------------------------
# Quick sanity check (no GPU or real model needed — mocked weights)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("relevance_scorer.py — sanity check with random / mocked tensors")
    print("=" * 65)

    torch.manual_seed(7)

    # -----------------------------------------------------------------------
    # CLIPRelevanceScorer — mock the CLIP projection matrix directly so we
    # do not need a GPU or real model weights.
    # -----------------------------------------------------------------------
    print()
    print("--- CLIPRelevanceScorer (mocked visual_proj) ---")

    N = 576       # number of CLIP patch tokens per image
    DV = 1024     # pre-connector CLIP vision hidden dim
    CLIP_DIM = 768  # joint CLIP embedding space dim

    # Mock ln_post / visual_proj so no open_clip weights are needed.
    mock_proj = torch.randn(DV, CLIP_DIM)
    scorer = CLIPRelevanceScorer.from_components(torch.nn.LayerNorm(DV), mock_proj)

    # Synthetic pre-connector features and question embedding.
    E_fake = torch.randn(N, DV)
    q_fake = F.normalize(torch.randn(CLIP_DIM), p=2, dim=0)  # pre-normalised (768,)

    R = scorer.compute(E_fake, q_fake)

    # Shape check.
    assert R.shape == (N,), f"Expected R shape ({N},), got {R.shape}"
    # Range check: cosine similarity is in [-1, 1].
    assert R.min().item() >= -1.0 - 1e-5, f"R.min()={R.min().item()} below -1"
    assert R.max().item() <=  1.0 + 1e-5, f"R.max()={R.max().item()} above +1"

    diag = scorer.diagnostics(R)
    assert set(diag.keys()) == {"min", "max", "mean", "std"}, (
        f"Unexpected diagnostics keys: {set(diag.keys())}"
    )

    print(f"  N={N}  DV={DV}  CLIP_DIM={CLIP_DIM}")
    print(f"  R.shape              : {tuple(R.shape)}  (expected ({N},))")
    print(f"  R range              : [{R.min():.4f}, {R.max():.4f}]  (expected in [-1, 1])")
    print(f"  diagnostics          : {diag}")
    print("  CLIPRelevanceScorer compute() — PASSED")

    # Test compute_batch with 3 images (use mocked encode_question).
    # We bypass encode_question by patching it.
    def _mock_encode_question(question: str) -> Tensor:
        return F.normalize(torch.randn(CLIP_DIM), p=2, dim=0)

    scorer._model = None  # not used in compute(); only used in encode_question
    scorer._tokenizer = None
    scorer.encode_question = _mock_encode_question  # type: ignore[method-assign]

    E_list = [torch.randn(N, DV) for _ in range(3)]
    questions = ["What colour is the dog?", "How many cars?", "Is it raining?"]
    R_batch = scorer.compute_batch(E_list, questions)

    assert len(R_batch) == 3, f"Expected 3 results, got {len(R_batch)}"
    for idx, r in enumerate(R_batch):
        assert r.shape == (N,), f"R_batch[{idx}] shape mismatch: {r.shape}"

    print(f"  compute_batch shapes : {[tuple(r.shape) for r in R_batch]}  — PASSED")

    # -----------------------------------------------------------------------
    # LLMEmbeddingRelevanceScorer — mock language model and tokenizer.
    # -----------------------------------------------------------------------
    print()
    print("--- LLMEmbeddingRelevanceScorer (mocked LM) ---")

    DT = 4096  # LLM hidden dim

    class _MockEmbedTokens(torch.nn.Module):
        """Minimal mock of an embedding table."""
        def __init__(self, vocab_size: int = 32000, embed_dim: int = DT) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(vocab_size, embed_dim), requires_grad=False
            )

        def forward(self, input_ids: Tensor) -> Tensor:  # (B, L) → (B, L, dt)
            return self.weight[input_ids]

    class _MockModel:
        def __init__(self) -> None:
            self.embed_tokens = _MockEmbedTokens()

    class _MockLM:
        def __init__(self) -> None:
            self.model = _MockModel()

    class _MockTokenizer:
        def __call__(self, text: str, **kwargs):
            # Return a fixed-length token id tensor.
            return {"input_ids": torch.randint(0, 32000, (1, 7))}

    llm_scorer = LLMEmbeddingRelevanceScorer(
        language_model=_MockLM(),
        tokenizer=_MockTokenizer(),
        device="cpu",
    )

    # Test encode_question.
    q_llm = llm_scorer.encode_question("What is the dog doing?")
    assert q_llm.shape == (DT,), f"q_llm shape mismatch: {q_llm.shape}"
    print(f"  q_llm.shape          : {tuple(q_llm.shape)}  (expected ({DT},))")

    # Test compute.
    T_fake = torch.randn(N, DT)
    R_llm = llm_scorer.compute(T_fake, q_llm)
    assert R_llm.shape == (N,), f"R_llm shape mismatch: {R_llm.shape}"
    print(f"  R_llm.shape          : {tuple(R_llm.shape)}  (expected ({N},))")

    # Numerical sanity: manual computation.
    expected_R_llm = (T_fake.float() @ q_llm.float()) / math.sqrt(DT)
    assert torch.allclose(R_llm, expected_R_llm, atol=1e-5), (
        "LLMEmbeddingRelevanceScorer.compute() output does not match manual calculation."
    )
    print("  LLMEmbeddingRelevanceScorer compute() — PASSED")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print()
    print("All relevance_scorer.py sanity checks PASSED.")
    sys.exit(0)
