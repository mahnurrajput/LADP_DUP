"""
relevance_scorer.py — LADP project
====================================
Implements two relevance scorers for the Loss-Aware Dynamic Projection (LADP)
pipeline.

CLIPRelevanceScorer  (FIXED architecture — primary scorer)
----------------------------------------------------------
Computes Rᵢ = cosine_sim(Eᵢ_proj, q_CLIP) where:
  - Eᵢ         : pre-connector CLIP vision features, shape (N, 1024)
                  output of CLIP ViT-L/14 at the penultimate hidden layer
  - Eᵢ_proj    : Eᵢ @ visual_proj  →  shape (N, 768) in the joint CLIP space
  - q_CLIP      : CLIP text encoder output for the question, shape (768,)
  - Both Eᵢ_proj and q_CLIP are L2-normalised before the dot product.

Why this avoids the modality gap:
  LLaVA-1.5 uses CLIP ViT-L/14 as its vision backbone.  The pre-connector
  features Eᵢ ARE CLIP vision features.  CLIP's visual.proj maps them into
  the shared image–text embedding space that CLIP was trained in.  The CLIP
  text encoder lives in the same space.  Cosine similarity in that space is
  therefore a principled relevance measure.

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

        # Extract the visual projection matrix.
        # In open_clip, for ViT-based models this lives at model.visual.proj
        # and has shape (dv, embed_dim) = (1024, 768) for ViT-L/14.
        if not hasattr(model.visual, "proj") or model.visual.proj is None:
            raise RuntimeError(
                f"open_clip model '{clip_model_name}' does not expose "
                "model.visual.proj.  Ensure you are using a ViT-based model "
                "with a linear projection head (e.g. ViT-L-14)."
            )

        # visual_proj: (dv, embed_dim) = (1024, 768)
        # Register as a buffer-like attribute so it moves with device calls.
        self._visual_proj: Tensor = model.visual.proj  # (1024, 768)

        # Keep only the text encoder and discard the rest of the vision tower
        # to save memory.  The text encoder is at model.token_embedding,
        # model.positional_embedding, model.transformer, and model.text_projection
        # in open_clip — it is simplest to keep the full model but only call
        # model.encode_text().
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(clip_model_name)

        logger.info(
            "CLIPRelevanceScorer ready.  visual_proj shape: %s",
            tuple(self._visual_proj.shape),
        )

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
        tokens: Tensor = self._tokenizer([question]).to(self._device)
        # encode_text returns (1, embed_dim) = (1, 768).
        q_raw: Tensor = self._model.encode_text(tokens)  # (1, 768)
        q_norm: Tensor = F.normalize(q_raw, p=2, dim=-1)  # (1, 768)
        return q_norm.squeeze(0)  # (768,)

    # ------------------------------------------------------------------
    # Per-image relevance scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute(self, E: Tensor, q_clip: Tensor) -> Tensor:
        """Compute per-patch relevance scores for one image.

        Parameters
        ----------
        E : Tensor, shape (N, 1024)
            Pre-connector CLIP vision features for N patches of one image.
            These should be RAW (non-normalised) hidden states from the CLIP
            vision tower penultimate layer, as returned by ``LlavaAdapter``.
        q_clip : Tensor, shape (768,) or (1, 768)
            L2-normalised CLIP text embedding of the question, as returned by
            :meth:`encode_question`.

        Returns
        -------
        Tensor of shape (N,)
            Cosine similarity Rᵢ for each patch i, in the range [-1, 1].

        Steps
        -----
        1. E_proj = E @ visual_proj                 → (N, 768)
        2. E_norm = L2_normalise(E_proj, dim=-1)    → (N, 768)
        3. q_norm = L2_normalise(q_clip, dim=-1)    → (1, 768)
        4. R      = (E_norm @ q_norm.T).squeeze(-1) → (N,)
        """
        # Ensure E is on the same device as the projection matrix.
        E = E.to(self._visual_proj.device)
        q_clip = q_clip.to(self._visual_proj.device)

        # Step 1: project pre-connector features into CLIP embedding space.
        E_proj: Tensor = E.float() @ self._visual_proj.float()  # (N, 768)

        # Step 2: L2-normalise the projected patches.
        E_norm: Tensor = F.normalize(E_proj, p=2, dim=-1)  # (N, 768)

        # Step 3: L2-normalise the question embedding (may already be, but
        #         re-normalising is a no-op if the vector already has unit norm).
        q_norm: Tensor = F.normalize(q_clip.view(1, -1).float(), p=2, dim=-1)  # (1, 768)

        # Step 4: Cosine similarities via dot products.
        R: Tensor = (E_norm @ q_norm.T).squeeze(-1)  # (N,)

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
            Each element is a Tensor of shape (N, 1024) — pre-connector
            features for one image.
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

    # Create a CLIPRelevanceScorer instance without calling __init__ so we
    # can inject a mock projection matrix without loading open_clip.
    scorer = object.__new__(CLIPRelevanceScorer)
    scorer._device = torch.device("cpu")
    scorer._clip_model_name = "ViT-L-14-336"
    scorer._pretrained = "openai"
    # Mock visual_proj: (1024, 768) random orthogonalish matrix.
    mock_proj = torch.randn(DV, CLIP_DIM)
    scorer._visual_proj = mock_proj

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
