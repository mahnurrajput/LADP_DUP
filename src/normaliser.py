"""
normaliser.py — LADP project
=============================
Provides ``Normaliser``, which maps raw per-patch loss L and relevance R
tensors into the [0, 1] range before they are combined by ``PriorityModule``.

Also provides the standalone ``validate_scores`` helper, which asserts that
a score tensor is finite and contained in [0, 1].

Design notes
------------
* Loss normalisation statistics (p5 / p95) are taken from ``NormStats`` fitted
  on the training split (robust mode) so that evaluation scores are calibrated
  against training-time extremes rather than the per-image range.
* Relevance normalisation is always per-image because relevance has no global
  training-split reference; it reflects how the query distributes attention
  across the patches of a single image.
* Both ``normalise_loss`` and ``normalise_relevance`` are pure functions of
  their inputs — no state is mutated and no model weights are involved.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import Tensor

from .loss_estimator import NormStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone validation helper
# ---------------------------------------------------------------------------


def validate_scores(scores: Tensor, name: str) -> bool:
    """Check that *scores* is a finite tensor with all values in [0, 1].

    Parameters
    ----------
    scores : Tensor
        1-D tensor of score values to validate.
    name : str
        Human-readable name used in error messages (e.g. ``"L_bar"``).

    Returns
    -------
    bool
        ``True`` if validation passes.

    Raises
    ------
    RuntimeError
        If any element of *scores* is ``NaN`` or ``Inf``.
    RuntimeError
        If any element of *scores* is outside the closed interval [0, 1].
    """
    if not torch.isfinite(scores).all():
        raise RuntimeError(
            f"validate_scores: '{name}' contains NaN or Inf values.  "
            f"Tensor stats — min: {scores.min().item()}, max: {scores.max().item()}, "
            f"any_nan: {scores.isnan().any().item()}, any_inf: {scores.isinf().any().item()}."
        )

    if scores.min().item() < 0.0 or scores.max().item() > 1.0:
        raise RuntimeError(
            f"validate_scores: '{name}' has values outside [0, 1].  "
            f"min={scores.min().item():.6f}, max={scores.max().item():.6f}."
        )

    return True


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


class Normaliser:
    """Normalises per-patch loss L and relevance R to [0, 1].

    Normalisation statistics for loss come from ``NormStats`` (fitted on the
    training split); relevance is normalised per-image because there is no
    global training reference for cosine similarity values.

    This class is stateless — both public methods are pure transformations of
    their inputs.  No model weights are stored or trained.
    """

    # ------------------------------------------------------------------
    # Loss normalisation
    # ------------------------------------------------------------------

    def normalise_loss(
        self,
        L: Tensor,
        norm_stats: NormStats,
        mode: Literal["robust", "per_image"] = "robust",
    ) -> Tensor:
        """Normalise raw per-patch loss to [0, 1].

        Parameters
        ----------
        L : Tensor, shape (N,)
            Raw per-patch reconstruction loss values as returned by
            ``LossEstimator.compute``.  Values are non-negative (MSE,
            norm-relative, or cosine distance).
        norm_stats : NormStats
            Fitted normalisation statistics from the training split.
            Only ``L_p5`` and ``L_p95`` are used in ``"robust"`` mode.
        mode : {"robust", "per_image"}
            Normalisation strategy:

            * ``"robust"`` (default) — uses global training-split percentiles:

              .. math::
                  \\bar{L}_i = \\operatorname{clip}\\!
                  \\left(\\frac{L_i - p_5}{p_{95} - p_5 + \\varepsilon},\\; 0,\\; 1\\right)

              Patches with loss below the 5th training percentile are mapped to
              0; those above the 95th are mapped to 1.  Values in between are
              linearly scaled.  This ensures evaluation scores are calibrated to
              training-time loss extremes.

            * ``"per_image"`` — min-max normalisation within this image's
              patches only:

              .. math::
                  \\bar{L}_i = \\frac{L_i - L_{\\min}}{L_{\\max} - L_{\\min} + \\varepsilon}

              Useful when no training statistics are available or when the
              absolute loss scale is irrelevant (ablation studies).

        Returns
        -------
        Tensor, shape (N,)
            Normalised loss values :math:`\\bar{L}` in [0, 1].

        Raises
        ------
        ValueError
            If *mode* is not one of the recognised strings.
        """
        if mode not in {"robust", "per_image"}:
            raise ValueError(
                f"normalise_loss: mode must be 'robust' or 'per_image'; got {mode!r}."
            )

        eps = 1e-8

        if mode == "robust":
            p5 = L.new_tensor(norm_stats.L_p5)    # scalar, same device as L
            p95 = L.new_tensor(norm_stats.L_p95)  # scalar, same device as L
            L_bar = (L - p5) / (p95 - p5 + eps)
            L_bar = L_bar.clamp(0.0, 1.0)

        else:  # per_image
            L_min = L.min()
            L_max = L.max()
            L_bar = (L - L_min) / (L_max - L_min + eps)

        logger.debug(
            "normalise_loss (mode=%s): L range [%.4f, %.4f] → L̄ range [%.4f, %.4f]",
            mode,
            L.min().item(),
            L.max().item(),
            L_bar.min().item(),
            L_bar.max().item(),
        )

        return L_bar  # (N,)

    # ------------------------------------------------------------------
    # Relevance normalisation
    # ------------------------------------------------------------------

    def normalise_relevance(
        self,
        R: Tensor,
        mode: Literal["minmax", "rank"] = "minmax",
    ) -> Tensor:
        """Normalise raw per-patch cosine similarity scores to [0, 1].

        Relevance scores from ``CLIPRelevanceScorer.compute`` are cosine
        similarities in [−1, 1].  This method maps them into [0, 1] so they
        can be combined with the normalised loss via ``PriorityModule``.

        Parameters
        ----------
        R : Tensor, shape (N,)
            Raw per-patch relevance scores.  For CLIP-space scoring these are
            cosine similarities in [−1, 1]; for LLM-space scoring they are
            scaled dot products that may fall outside that range.
        mode : {"minmax", "rank"}
            Normalisation strategy:

            * ``"minmax"`` (default) — linear scaling to [0, 1]:

              .. math::
                  \\bar{R}_i = \\frac{R_i - R_{\\min}}{R_{\\max} - R_{\\min} + \\varepsilon}

              The patch with the lowest cosine similarity maps to 0; the one
              with the highest maps to 1.

            * ``"rank"`` — rank-based normalisation:

              .. math::
                  \\bar{R}_i = \\frac{\\operatorname{rank}(R_i)}{N - 1}

              where ranks run from 0 (smallest) to N − 1 (largest).  This
              strategy is robust to outlier relevance values and produces a
              uniform distribution of scores regardless of the raw score
              distribution.  When N = 1 the single patch receives a score of 0.

        Returns
        -------
        Tensor, shape (N,)
            Normalised relevance values :math:`\\bar{R}` in [0, 1].

        Raises
        ------
        ValueError
            If *mode* is not one of the recognised strings.
        """
        if mode not in {"minmax", "rank"}:
            raise ValueError(
                f"normalise_relevance: mode must be 'minmax' or 'rank'; got {mode!r}."
            )

        eps = 1e-8

        if mode == "minmax":
            R_min = R.min()
            R_max = R.max()
            R_bar = (R - R_min) / (R_max - R_min + eps)

        else:  # rank
            N = R.shape[0]
            if N == 1:
                # A single patch has rank 0 by convention.
                R_bar = R.new_zeros(1)
            else:
                # argsort twice gives the dense rank of each element.
                # First argsort: indices that would sort R ascending.
                # Second argsort: for each element, its rank in ascending order.
                sorted_indices = torch.argsort(R)         # positions of values in sorted order
                ranks = torch.argsort(sorted_indices)     # rank of each original element
                R_bar = ranks.float() / float(N - 1)     # (N,) in [0, 1]

        logger.debug(
            "normalise_relevance (mode=%s): R range [%.4f, %.4f] → R̄ range [%.4f, %.4f]",
            mode,
            R.min().item(),
            R.max().item(),
            R_bar.min().item(),
            R_bar.max().item(),
        )

        return R_bar  # (N,)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("LADP normaliser.py — shape and value sanity checks")
    print("=" * 60)

    torch.manual_seed(0)
    N = 576  # typical number of CLIP patches

    # Synthetic NormStats (matches loss_estimator.py conventions).
    from loss_estimator import NormStats as _NS  # noqa: PLC0415
    DV, DT = 1024, 4096
    ns = _NS(
        E_mean=torch.zeros(DV),
        E_std=torch.ones(DV),
        T_mean=torch.zeros(DT),
        T_std=torch.ones(DT),
        L_p5=0.05,
        L_p95=0.95,
    )

    normaliser = Normaliser()
    L_raw = torch.rand(N) * 2.0  # values in [0, 2] — some outside [p5, p95]

    # --- Loss normalisation: robust ---
    L_bar_robust = normaliser.normalise_loss(L_raw, ns, mode="robust")
    assert L_bar_robust.shape == (N,), f"Shape mismatch: {L_bar_robust.shape}"
    assert L_bar_robust.min().item() >= 0.0
    assert L_bar_robust.max().item() <= 1.0
    print(f"  Loss robust:    L̄ in [{L_bar_robust.min():.4f}, {L_bar_robust.max():.4f}]  — PASSED")

    # --- Loss normalisation: per_image ---
    L_bar_pi = normaliser.normalise_loss(L_raw, ns, mode="per_image")
    assert L_bar_pi.shape == (N,)
    assert L_bar_pi.min().item() >= 0.0
    assert L_bar_pi.max().item() <= 1.0 + 1e-6
    print(f"  Loss per_image: L̄ in [{L_bar_pi.min():.4f}, {L_bar_pi.max():.4f}]  — PASSED")

    # --- Relevance normalisation: minmax ---
    R_raw = torch.randn(N)  # cosine sims in roughly [-3, 3] for unit Gaussians
    R_bar_mm = normaliser.normalise_relevance(R_raw, mode="minmax")
    assert R_bar_mm.shape == (N,)
    assert R_bar_mm.min().item() >= 0.0
    assert R_bar_mm.max().item() <= 1.0 + 1e-6
    print(f"  Relevance minmax: R̄ in [{R_bar_mm.min():.4f}, {R_bar_mm.max():.4f}]  — PASSED")

    # --- Relevance normalisation: rank ---
    R_bar_rank = normaliser.normalise_relevance(R_raw, mode="rank")
    assert R_bar_rank.shape == (N,)
    assert R_bar_rank.min().item() >= 0.0
    assert R_bar_rank.max().item() <= 1.0 + 1e-6
    # Rank scores should be exactly {0/(N-1), 1/(N-1), …, 1}
    expected_max = 1.0
    assert abs(R_bar_rank.max().item() - expected_max) < 1e-5
    print(f"  Relevance rank:   R̄ in [{R_bar_rank.min():.4f}, {R_bar_rank.max():.4f}]  — PASSED")

    # --- Single-patch edge case for rank mode ---
    R_single = torch.tensor([0.42])
    R_bar_single = normaliser.normalise_relevance(R_single, mode="rank")
    assert R_bar_single.item() == 0.0, f"Single-patch rank should be 0.0, got {R_bar_single.item()}"
    print(f"  Relevance rank (N=1): R̄={R_bar_single.item():.4f}  — PASSED")

    # --- validate_scores: valid tensor ---
    valid_tensor = torch.rand(N)
    assert validate_scores(valid_tensor, "test_valid") is True
    print("  validate_scores (valid):   PASSED")

    # --- validate_scores: NaN/Inf should raise ---
    nan_tensor = torch.rand(N)
    nan_tensor[5] = float("nan")
    try:
        validate_scores(nan_tensor, "nan_test")
        print("  FAIL: expected RuntimeError for NaN tensor")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"  validate_scores (NaN):     PASSED — {exc}")

    inf_tensor = torch.rand(N)
    inf_tensor[0] = float("inf")
    try:
        validate_scores(inf_tensor, "inf_test")
        print("  FAIL: expected RuntimeError for Inf tensor")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"  validate_scores (Inf):     PASSED — {exc}")

    # --- validate_scores: out-of-range should raise ---
    oob_tensor = torch.rand(N)
    oob_tensor[10] = 1.5
    try:
        validate_scores(oob_tensor, "oob_test")
        print("  FAIL: expected RuntimeError for out-of-range tensor")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"  validate_scores (OOB):     PASSED — {exc}")

    print("\nAll normaliser.py sanity checks passed.")
