"""
priority_module.py — LADP project
===================================
Combines normalised per-patch loss :math:`\\bar{L}` and relevance
:math:`\\bar{R}` into a scalar priority score S that is used to select
which visual patches are presented to the LLM.

Both inputs must already be in [0, 1] (see ``normaliser.Normaliser``).

Combination modes
-----------------
additive
    :math:`S_i = \\alpha \\bar{L}_i + \\beta \\bar{R}_i`

    with the constraint :math:`\\alpha + \\beta = 1`,
    :math:`\\alpha, \\beta \\geq 0`.

    * :math:`\\alpha = 1, \\beta = 0` → loss-only ablation.
    * :math:`\\alpha = 0, \\beta = 1` → relevance-only ablation.
    * :math:`\\alpha = \\beta = 0.5` → equal weighting (default).

multiplicative
    :math:`S_i = \\bar{L}_i \\cdot \\bar{R}_i`

    Both high loss *and* high relevance must hold simultaneously for a patch
    to receive a high priority score.  This can suppress patches that are
    either easy (low loss) or off-topic (low relevance) more aggressively
    than the additive combination.

Patch selection
---------------
After computing S, patches are ranked in descending order (highest priority
first).  ``top_k_indices`` then returns the top-k indices in *ascending*
order so that the spatial / sequential ordering of patches is preserved for
the LLM's context window.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_VALID_MODES = {"additive", "multiplicative"}


class PriorityModule:
    """Combine normalised loss and relevance into a per-patch priority score.

    Parameters
    ----------
    mode : {"additive", "multiplicative"}
        Combination strategy.  See module docstring for details.
        Default is ``"additive"``.
    alpha : float
        Weight for the normalised loss :math:`\\bar{L}`.  Must be non-negative.
        For ``"additive"`` mode, ``alpha + beta`` must equal 1 (within 1e-6).
        Default 0.5.
    beta : float
        Weight for the normalised relevance :math:`\\bar{R}`.  Must be
        non-negative.  For ``"additive"`` mode, ``alpha + beta`` must equal 1
        (within 1e-6).  Default 0.5.

    Raises
    ------
    ValueError
        If ``mode`` is not in ``{"additive", "multiplicative"}``.
    ValueError
        If ``alpha`` or ``beta`` is negative.
    ValueError
        If ``mode == "additive"`` and ``|alpha + beta − 1| > 1e-6``.
    """

    def __init__(
        self,
        mode: str = "additive",
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        # --- validate mode -----------------------------------------------
        if mode not in _VALID_MODES:
            raise ValueError(
                f"PriorityModule: mode must be one of {_VALID_MODES}; got {mode!r}."
            )

        # --- validate weights --------------------------------------------
        if alpha < 0.0:
            raise ValueError(
                f"PriorityModule: alpha must be >= 0; got alpha={alpha}."
            )
        if beta < 0.0:
            raise ValueError(
                f"PriorityModule: beta must be >= 0; got beta={beta}."
            )

        if mode == "additive":
            if abs(alpha + beta - 1.0) > 1e-6:
                raise ValueError(
                    f"PriorityModule: for mode='additive', alpha + beta must equal 1.0 "
                    f"(within 1e-6 tolerance); got alpha={alpha}, beta={beta}, "
                    f"alpha+beta={alpha + beta}."
                )

        self.mode = mode
        self.alpha = alpha
        self.beta = beta

        logger.debug(
            "PriorityModule initialised: mode=%s, alpha=%.4f, beta=%.4f",
            self.mode,
            self.alpha,
            self.beta,
        )

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def forward(self, L_bar: Tensor, R_bar: Tensor) -> Tensor:
        """Compute per-patch priority scores.

        Parameters
        ----------
        L_bar : Tensor, shape (N,)
            Normalised per-patch reconstruction loss in [0, 1], as produced by
            ``Normaliser.normalise_loss``.
        R_bar : Tensor, shape (N,)
            Normalised per-patch relevance in [0, 1], as produced by
            ``Normaliser.normalise_relevance``.

        Returns
        -------
        S : Tensor, shape (N,)
            Priority scores in [0, 1].

        Raises
        ------
        ValueError
            If ``L_bar`` and ``R_bar`` do not have the same shape.

        Notes
        -----
        For ``"additive"`` mode, since :math:`\\alpha + \\beta = 1` and both
        inputs are in [0, 1], the output is guaranteed to be in [0, 1].

        For ``"multiplicative"`` mode, the product of two values in [0, 1]
        is also in [0, 1].
        """
        if L_bar.shape != R_bar.shape:
            raise ValueError(
                f"PriorityModule.forward: L_bar and R_bar must have the same shape.  "
                f"Got L_bar.shape={tuple(L_bar.shape)}, R_bar.shape={tuple(R_bar.shape)}."
            )

        if self.mode == "additive":
            S = self.alpha * L_bar + self.beta * R_bar
        else:  # multiplicative
            S = L_bar * R_bar

        logger.debug(
            "PriorityModule.forward (mode=%s): S range [%.4f, %.4f]",
            self.mode,
            S.min().item(),
            S.max().item(),
        )

        return S  # (N,)

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    def rank(self, S: Tensor) -> Tensor:
        """Return indices that sort *S* in descending order.

        The patch with the **highest** priority score appears first
        (index 0 in the returned tensor).

        Parameters
        ----------
        S : Tensor, shape (N,)
            Priority scores, e.g. the output of :meth:`forward`.

        Returns
        -------
        Tensor, shape (N,), dtype long
            Indices such that ``S[rank(S)]`` is sorted in descending order.
        """
        return torch.argsort(S, descending=True)

    def top_k_indices(self, S: Tensor, k: int) -> Tensor:
        """Return the top-*k* patch indices sorted in **ascending** order.

        Ascending order preserves the original spatial / sequential ordering of
        patches in the image, which is important for the LLM's context window
        because position matters for attention.

        Parameters
        ----------
        S : Tensor, shape (N,)
            Priority scores, e.g. the output of :meth:`forward`.
        k : int
            Number of top-priority patches to select.  Must satisfy
            ``1 <= k <= N``.

        Returns
        -------
        Tensor, shape (k,), dtype long
            The *k* indices with the highest priority scores, returned in
            ascending (original spatial) order.

        Raises
        ------
        ValueError
            If ``k < 1`` or ``k > N``.
        """
        N = S.shape[0]
        if k < 1:
            raise ValueError(
                f"PriorityModule.top_k_indices: k must be >= 1; got k={k}."
            )
        if k > N:
            raise ValueError(
                f"PriorityModule.top_k_indices: k={k} exceeds the number of patches N={N}."
            )

        if k == N:
            # All patches selected — just return sorted ascending indices.
            return torch.arange(N, device=S.device, dtype=torch.long)

        # torch.topk returns values and indices of the k largest elements.
        _, top_indices = torch.topk(S, k, largest=True, sorted=False)

        # Sort ascending to restore spatial ordering.
        top_indices_sorted, _ = torch.sort(top_indices)

        logger.debug(
            "PriorityModule.top_k_indices: k=%d / N=%d  indices range [%d, %d]",
            k,
            N,
            int(top_indices_sorted.min().item()),
            int(top_indices_sorted.max().item()),
        )

        return top_indices_sorted  # (k,)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.mode == "additive":
            formula = f"S = {self.alpha:.4f}·L̄ + {self.beta:.4f}·R̄"
        else:
            formula = "S = L̄ · R̄"
        return (
            f"PriorityModule(mode={self.mode!r}, alpha={self.alpha}, "
            f"beta={self.beta})  [{formula}]"
        )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("LADP priority_module.py — sanity checks")
    print("=" * 60)

    torch.manual_seed(42)
    N = 576  # typical patch count

    L_bar = torch.rand(N)
    R_bar = torch.rand(N)

    # ------------------------------------------------------------------
    print("\n[1] Additive mode (alpha=0.5, beta=0.5)")
    pm_add = PriorityModule(mode="additive", alpha=0.5, beta=0.5)
    print(f"  {pm_add}")
    S_add = pm_add.forward(L_bar, R_bar)
    assert S_add.shape == (N,), f"Shape mismatch: {S_add.shape}"
    assert S_add.min().item() >= 0.0 and S_add.max().item() <= 1.0 + 1e-6, (
        f"S_add out of [0,1]: [{S_add.min():.4f}, {S_add.max():.4f}]"
    )
    print(f"  S_add range: [{S_add.min():.4f}, {S_add.max():.4f}]  — PASSED")

    # ------------------------------------------------------------------
    print("\n[2] Additive mode — loss-only ablation (alpha=1, beta=0)")
    pm_loss_only = PriorityModule(mode="additive", alpha=1.0, beta=0.0)
    S_loss = pm_loss_only.forward(L_bar, R_bar)
    assert torch.allclose(S_loss, L_bar), "Loss-only: S should equal L_bar"
    print("  S == L_bar  — PASSED")

    # ------------------------------------------------------------------
    print("\n[3] Additive mode — relevance-only ablation (alpha=0, beta=1)")
    pm_rel_only = PriorityModule(mode="additive", alpha=0.0, beta=1.0)
    S_rel = pm_rel_only.forward(L_bar, R_bar)
    assert torch.allclose(S_rel, R_bar), "Relevance-only: S should equal R_bar"
    print("  S == R_bar  — PASSED")

    # ------------------------------------------------------------------
    print("\n[4] Multiplicative mode")
    pm_mul = PriorityModule(mode="multiplicative")
    print(f"  {pm_mul}")
    S_mul = pm_mul.forward(L_bar, R_bar)
    assert S_mul.shape == (N,)
    assert S_mul.min().item() >= 0.0 and S_mul.max().item() <= 1.0 + 1e-6
    expected_mul = L_bar * R_bar
    assert torch.allclose(S_mul, expected_mul)
    print(f"  S_mul range: [{S_mul.min():.4f}, {S_mul.max():.4f}]  — PASSED")

    # ------------------------------------------------------------------
    print("\n[5] rank() — descending order")
    S_test = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7])
    ranked = pm_add.rank(S_test)
    assert ranked.tolist() == [1, 4, 2, 3, 0], (
        f"Expected [1, 4, 2, 3, 0], got {ranked.tolist()}"
    )
    print(f"  rank({S_test.tolist()}) = {ranked.tolist()}  — PASSED")

    # ------------------------------------------------------------------
    print("\n[6] top_k_indices() — ascending spatial order")
    k = 3
    top_k = pm_add.top_k_indices(S_test, k=k)
    assert top_k.shape == (k,), f"Shape mismatch: {top_k.shape}"
    # Top-3 by value: indices 1 (0.9), 4 (0.7), 2 (0.5) → sorted ascending: [1, 2, 4]
    assert top_k.tolist() == [1, 2, 4], (
        f"Expected [1, 2, 4], got {top_k.tolist()}"
    )
    print(f"  top_{k}_indices({S_test.tolist()}) = {top_k.tolist()}  — PASSED")

    # k == N edge case
    top_all = pm_add.top_k_indices(S_test, k=len(S_test))
    assert top_all.tolist() == list(range(len(S_test)))
    print(f"  top_k_indices (k=N): {top_all.tolist()}  — PASSED")

    # ------------------------------------------------------------------
    print("\n[7] Constructor validation errors")

    for bad_kwargs, expected_msg_fragment in [
        ({"mode": "geometric"}, "mode must be one of"),
        ({"mode": "additive", "alpha": -0.1, "beta": 1.1}, "alpha must be"),
        ({"mode": "additive", "alpha": 0.5, "beta": -0.1}, "beta must be"),
        ({"mode": "additive", "alpha": 0.3, "beta": 0.3}, "alpha + beta must equal 1.0"),
    ]:
        try:
            PriorityModule(**bad_kwargs)
            print(f"  FAIL: expected ValueError for {bad_kwargs}")
            sys.exit(1)
        except ValueError as exc:
            assert expected_msg_fragment in str(exc), (
                f"Wrong error message: {exc}"
            )
            print(f"  ValueError for {bad_kwargs}  — PASSED")

    # ------------------------------------------------------------------
    print("\n[8] forward() shape mismatch")
    try:
        pm_add.forward(torch.rand(10), torch.rand(5))
        print("  FAIL: expected ValueError for shape mismatch")
        sys.exit(1)
    except ValueError as exc:
        print(f"  ValueError (shape mismatch)  — PASSED — {exc}")

    # ------------------------------------------------------------------
    print("\n[9] top_k_indices() boundary errors")
    for bad_k, expected_msg in [(0, "k must be >= 1"), (N + 1, "exceeds the number of patches")]:
        try:
            pm_add.top_k_indices(S_add, k=bad_k)
            print(f"  FAIL: expected ValueError for k={bad_k}")
            sys.exit(1)
        except ValueError as exc:
            print(f"  ValueError (k={bad_k})  — PASSED — {exc}")

    print("\nAll priority_module.py sanity checks passed.")
