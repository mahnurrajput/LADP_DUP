"""
knor.py — LADP diagnostics
===========================
Implements the KNOR (k-Nearest Neighbours Overlap Ratio) diagnostic from the
paper "Lost in Embeddings".

**What KNOR measures:**

For each image x:
    - N_k_before(x): k nearest neighbours in PRE-connector feature space
                     ψ(x) = mean-pooled E  (mean over patch dimension → R^dv)
    - N_k_after(x):  k nearest neighbours in POST-connector feature space
                     C(ψ(x)) = mean-pooled T  (mean over patch dimension → R^dt)
    - KNOR(x, k) = |N_k_before(x) ∩ N_k_after(x)| / k
    - KNOR(k)    = mean over all images

If KNOR < 1.0, the connector changes the neighbourhood structure, which the
paper interprets as evidence of information loss in the visual pathway.

Also contains ``correlate_loss_with_error``, which links per-image patch loss
(from the LADP reconstruction objective) to VQA answer correctness via Pearson
and Spearman correlation.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KNOR diagnostic
# ---------------------------------------------------------------------------


class KNORDiagnostic:
    """Compute the k-Nearest Neighbours Overlap Ratio between pre- and
    post-connector feature spaces.

    Parameters
    ----------
    k : int
        Neighbourhood size.  Must be strictly positive and less than the
        total number of images in the dataset.
    """

    def __init__(self, k: int = 10) -> None:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")
        self.k = k

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        E_pooled: Tensor,  # (num_images, dv) — mean-pooled pre-connector
        T_pooled: Tensor,  # (num_images, dt) — mean-pooled post-connector
    ) -> dict:
        """Compute KNOR for each image and return aggregated statistics.

        Parameters
        ----------
        E_pooled : Tensor, shape (num_images, dv)
            Mean-pooled PRE-connector CLIP vision features.  One vector per
            image, obtained by averaging over the N patch dimension of E.
        T_pooled : Tensor, shape (num_images, dt)
            Mean-pooled POST-connector projected tokens.  One vector per
            image, obtained by averaging over the N patch dimension of T.

        Returns
        -------
        dict with keys:
            ``"knor_per_image"`` — Tensor of shape (num_images,), the
                individual KNOR(x, k) score for each image.
            ``"knor_mean"``      — float, mean KNOR across all images.
            ``"knor_std"``       — float, standard deviation of KNOR.
            ``"k"``              — int, the neighbourhood size used.

        Raises
        ------
        ValueError
            If the number of images is not the same in both tensors, or if
            k >= num_images (neighbourhood cannot be formed).
        """
        num_images_E = E_pooled.shape[0]
        num_images_T = T_pooled.shape[0]

        if num_images_E != num_images_T:
            raise ValueError(
                f"E_pooled and T_pooled must have the same number of images.  "
                f"Got E_pooled.shape[0]={num_images_E} and "
                f"T_pooled.shape[0]={num_images_T}."
            )

        num_images = num_images_E

        if self.k >= num_images:
            raise ValueError(
                f"k={self.k} must be strictly less than the total number of "
                f"images ({num_images}), because each image cannot be its own "
                "neighbour."
            )

        # Move to the same device; keep computation on CPU if inputs are CPU.
        device = E_pooled.device

        # ------------------------------------------------------------------
        # Step 1: L2-normalise both feature matrices so that the dot product
        #         equals cosine similarity.
        # ------------------------------------------------------------------
        E_norm: Tensor = F.normalize(E_pooled.float(), p=2, dim=-1)  # (M, dv)
        T_norm: Tensor = F.normalize(T_pooled.float(), p=2, dim=-1)  # (M, dt)

        # ------------------------------------------------------------------
        # Step 2: Compute full pairwise cosine similarity matrices.
        #         sim_E[i, j] = cosine_sim(E_norm[i], E_norm[j])
        # For large M this is O(M² · d).  For typical diagnostic runs with
        # M ≤ 10,000 and d ≤ 4096 this is feasible.
        # ------------------------------------------------------------------
        sim_E: Tensor = E_norm @ E_norm.T  # (M, M)
        sim_T: Tensor = T_norm @ T_norm.T  # (M, M)

        # ------------------------------------------------------------------
        # Step 3: For each image i, find the top-(k+1) similarity indices
        #         (the +1 is because index i itself will appear at rank 0
        #         with similarity ≈ 1.0 and must be excluded).
        # ------------------------------------------------------------------
        # topk returns values and indices of the k+1 largest entries per row.
        _, top_indices_E = sim_E.topk(self.k + 1, dim=-1, largest=True, sorted=True)
        # top_indices_E: (M, k+1)

        _, top_indices_T = sim_T.topk(self.k + 1, dim=-1, largest=True, sorted=True)
        # top_indices_T: (M, k+1)

        # ------------------------------------------------------------------
        # Step 4: Remove self (index i) from each row.  Since similarities
        #         are computed with L2-normalised vectors, the self-similarity
        #         is exactly 1.0, which is the maximum.  It therefore always
        #         occupies position 0 in the sorted indices.
        # ------------------------------------------------------------------
        # We drop column 0 (the self-index) → (M, k)
        nbrs_E: Tensor = top_indices_E[:, 1:]  # (M, k)
        nbrs_T: Tensor = top_indices_T[:, 1:]  # (M, k)

        # ------------------------------------------------------------------
        # Step 5: Compute intersection size |N_k_before ∩ N_k_after| / k.
        #
        # Efficient vectorised approach: for each row, check which columns of
        # nbrs_T appear in nbrs_E using broadcasting.
        #   match[i, a, b] = 1 iff nbrs_T[i, a] == nbrs_E[i, b]
        # Then any(dim=-1) gives whether each T-neighbour is also an E-neighbour.
        # ------------------------------------------------------------------
        # nbrs_E: (M, k)  → unsqueeze → (M, 1, k)
        # nbrs_T: (M, k)  → unsqueeze → (M, k, 1)
        match: Tensor = (
            nbrs_T.unsqueeze(-1) == nbrs_E.unsqueeze(-2)
        )  # (M, k, k) bool

        # For each row i and T-neighbour position a, check if it matches any
        # E-neighbour position b.
        in_both: Tensor = match.any(dim=-1)  # (M, k) bool

        # Intersection size per image, then divide by k.
        intersection_sizes: Tensor = in_both.sum(dim=-1).float()  # (M,)
        knor_per_image: Tensor = intersection_sizes / self.k        # (M,)

        return {
            "knor_per_image": knor_per_image,
            "knor_mean": float(knor_per_image.mean().item()),
            "knor_std": float(knor_per_image.std().item()),
            "k": self.k,
        }

    # ------------------------------------------------------------------
    # Cache-based computation
    # ------------------------------------------------------------------

    def compute_from_cache(self, cache_dir: str) -> dict:
        """Load cached feature files and compute KNOR.

        Each file is expected at ``cache_dir/features_{i}.pt`` and must
        contain a dict with keys:
            ``"E"`` — Tensor of shape ``(N, dv)`` — pre-connector patch
                      features for one image.
            ``"T"`` — Tensor of shape ``(N, dt)`` — post-connector tokens
                      for the same image.

        The features are mean-pooled over the patch dimension (N) before
        calling :meth:`compute`.

        Parameters
        ----------
        cache_dir : str
            Directory containing ``features_{i}.pt`` files.

        Returns
        -------
        dict
            Same structure as :meth:`compute`.

        Raises
        ------
        FileNotFoundError
            If no ``features_*.pt`` files are found in ``cache_dir``.
        KeyError
            If a cache file is missing the ``"E"`` or ``"T"`` key.
        """
        pattern = os.path.join(cache_dir, "features_*.pt")
        file_paths = sorted(glob.glob(pattern))

        if not file_paths:
            raise FileNotFoundError(
                f"No feature cache files matching '{pattern}' found.  "
                "Run the LADP feature extraction pipeline first."
            )

        logger.info(
            "Loading %d cached feature file(s) from '%s' …",
            len(file_paths),
            cache_dir,
        )

        E_pooled_list: List[Tensor] = []
        T_pooled_list: List[Tensor] = []

        for path in file_paths:
            data: dict = torch.load(path, map_location="cpu", weights_only=True)

            if "E" not in data:
                raise KeyError(
                    f"Cache file '{path}' is missing the required key 'E'."
                )
            if "T" not in data:
                raise KeyError(
                    f"Cache file '{path}' is missing the required key 'T'."
                )

            E: Tensor = data["E"]  # (N, dv)
            T: Tensor = data["T"]  # (N, dt)

            # Mean-pool over the patch (sequence) dimension.
            E_pooled_list.append(E.mean(dim=0))  # (dv,)
            T_pooled_list.append(T.mean(dim=0))  # (dt,)

        # Stack into matrices: (num_images, dv) and (num_images, dt).
        E_pooled: Tensor = torch.stack(E_pooled_list, dim=0)
        T_pooled: Tensor = torch.stack(T_pooled_list, dim=0)

        logger.info(
            "Pooled feature matrices ready.  E_pooled=%s  T_pooled=%s",
            tuple(E_pooled.shape),
            tuple(T_pooled.shape),
        )

        return self.compute(E_pooled, T_pooled)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self, results: dict) -> str:
        """Return a human-readable summary string for logging.

        Parameters
        ----------
        results : dict
            Output of :meth:`compute` or :meth:`compute_from_cache`.

        Returns
        -------
        str
            Multi-line summary string.
        """
        knor_mean: float = results["knor_mean"]
        knor_std: float = results["knor_std"]
        k: int = results["k"]

        # Diagnosis based on the paper's threshold (KNOR = 1.0 means perfect
        # neighbourhood preservation; any lower indicates connector-induced
        # information loss).
        if knor_mean >= 0.95:
            diagnosis = "neighbourhood structure largely preserved (low information loss risk)"
        elif knor_mean >= 0.70:
            diagnosis = "moderate neighbourhood disruption — connector may cause information loss"
        else:
            diagnosis = "SEVERE neighbourhood disruption — connector causes significant information loss"

        lines = [
            "=" * 60,
            "KNOR Diagnostic Summary",
            "=" * 60,
            f"  Neighbourhood size k : {k}",
            f"  KNOR mean            : {knor_mean:.4f}",
            f"  KNOR std             : {knor_std:.4f}",
            f"  Diagnosis            : {diagnosis}",
            "",
            "  Interpretation: KNOR = 1.0 means every k-NN in pre-connector",
            "  space is also a k-NN in post-connector space (perfect preservation).",
            "  KNOR < 1.0 indicates neighbourhood reshuffling by the connector,",
            "  which 'Lost in Embeddings' associates with visual information loss.",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loss-correctness correlation
# ---------------------------------------------------------------------------


def correlate_loss_with_error(
    losses: list[float],  # mean patch loss per image
    correct: list[bool],  # whether VQA answer is correct for that image
) -> dict:
    """Compute correlation between mean patch loss and VQA answer incorrectness.

    The hypothesis from the LADP paper is that images where the LADP
    reconstruction loss is high correspond to cases where the connector
    discards visually relevant information — which in turn should produce
    incorrect VQA answers.

    This function tests that hypothesis by measuring the correlation between
    the per-image mean patch reconstruction loss and the binary indicator
    of incorrect answering (1 = wrong, 0 = correct).

    Parameters
    ----------
    losses : list of float
        Per-image mean patch reconstruction loss values.  Length M.
    correct : list of bool
        Per-image VQA correctness flags.  ``True`` means the model produced
        the correct answer; ``False`` means it was wrong.  Length M.

    Returns
    -------
    dict with keys:
        ``"pearson_r"``       — float, Pearson correlation coefficient.
        ``"pearson_p"``       — float, two-tailed p-value for Pearson r.
        ``"spearman_r"``      — float, Spearman rank correlation.
        ``"spearman_p"``      — float, two-tailed p-value for Spearman r.
        ``"mean_loss_correct"`` — float, mean loss on correctly answered images.
        ``"mean_loss_wrong"``   — float, mean loss on incorrectly answered images.

    Raises
    ------
    ValueError
        If ``losses`` and ``correct`` have different lengths, or if fewer
        than 3 samples are provided (Pearson / Spearman are undefined).
    """
    if len(losses) != len(correct):
        raise ValueError(
            f"losses and correct must have the same length.  "
            f"Got len(losses)={len(losses)} and len(correct)={len(correct)}."
        )

    if len(losses) < 3:
        raise ValueError(
            f"At least 3 samples are required to compute correlations.  "
            f"Got {len(losses)}."
        )

    # Convert incorrectness: 1 if wrong (not correct), 0 if right.
    errors: list[int] = [0 if c else 1 for c in correct]

    # Pearson correlation.
    pearson_r, pearson_p = scipy_stats.pearsonr(losses, errors)

    # Spearman rank correlation.
    spearman_r, spearman_p = scipy_stats.spearmanr(losses, errors)

    # Stratified means.
    correct_losses = [l for l, c in zip(losses, correct) if c]
    wrong_losses = [l for l, c in zip(losses, correct) if not c]

    mean_loss_correct = float(sum(correct_losses) / len(correct_losses)) if correct_losses else float("nan")
    mean_loss_wrong = float(sum(wrong_losses) / len(wrong_losses)) if wrong_losses else float("nan")

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "mean_loss_correct": mean_loss_correct,
        "mean_loss_wrong": mean_loss_wrong,
    }


# ---------------------------------------------------------------------------
# Quick sanity check (no GPU or real model needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("knor.py — sanity check with random tensors")
    print("=" * 60)

    torch.manual_seed(42)

    NUM_IMAGES = 50
    DV = 1024   # pre-connector CLIP hidden dim
    DT = 4096   # post-connector LLM hidden dim
    K = 10

    # Synthetic mean-pooled feature matrices.
    E_pooled = torch.randn(NUM_IMAGES, DV)
    T_pooled = torch.randn(NUM_IMAGES, DT)

    diag = KNORDiagnostic(k=K)
    results = diag.compute(E_pooled, T_pooled)

    # --- shape checks ---
    assert results["knor_per_image"].shape == (NUM_IMAGES,), (
        f"knor_per_image shape mismatch: {results['knor_per_image'].shape}"
    )
    assert isinstance(results["knor_mean"], float), "knor_mean must be a float"
    assert isinstance(results["knor_std"], float), "knor_std must be a float"
    assert results["k"] == K, f"k mismatch: {results['k']} != {K}"
    assert 0.0 <= results["knor_mean"] <= 1.0, (
        f"knor_mean out of [0,1] range: {results['knor_mean']}"
    )

    print(f"  NUM_IMAGES : {NUM_IMAGES}")
    print(f"  DV         : {DV}  DT : {DT}  k : {K}")
    print(f"  knor_per_image shape : {tuple(results['knor_per_image'].shape)}")
    print(f"  knor_mean  : {results['knor_mean']:.4f}  (expected ~1/k={1/K:.4f} for random)")
    print(f"  knor_std   : {results['knor_std']:.4f}")
    print()

    # Verify that identical E and T give KNOR = 1.0.
    same_pooled = torch.randn(NUM_IMAGES, DV)
    results_same = diag.compute(same_pooled, same_pooled)
    assert abs(results_same["knor_mean"] - 1.0) < 1e-5, (
        f"Expected KNOR=1.0 when E_pooled == T_pooled, got {results_same['knor_mean']}"
    )
    print("  KNOR=1.0 check (E_pooled == T_pooled): PASSED")

    # Print summary.
    print()
    print(diag.summary(results))

    # --- correlate_loss_with_error ---
    print()
    print("-" * 60)
    print("correlate_loss_with_error — sanity check")
    print("-" * 60)

    import random
    random.seed(0)

    # Simulate: higher loss → more likely to be wrong.
    losses_sim = [random.uniform(0.0, 2.0) for _ in range(100)]
    # Answers are more likely wrong when loss > 1.0.
    correct_sim = [random.random() > (l / 2.0) for l in losses_sim]

    corr = correlate_loss_with_error(losses_sim, correct_sim)

    assert "pearson_r" in corr
    assert "spearman_r" in corr
    assert "mean_loss_correct" in corr
    assert "mean_loss_wrong" in corr
    assert -1.0 <= corr["pearson_r"] <= 1.0, f"pearson_r out of range: {corr['pearson_r']}"
    assert -1.0 <= corr["spearman_r"] <= 1.0, f"spearman_r out of range: {corr['spearman_r']}"

    print(f"  Samples           : {len(losses_sim)}")
    print(f"  pearson_r         : {corr['pearson_r']:.4f}  (p={corr['pearson_p']:.4f})")
    print(f"  spearman_r        : {corr['spearman_r']:.4f}  (p={corr['spearman_p']:.4f})")
    print(f"  mean_loss_correct : {corr['mean_loss_correct']:.4f}")
    print(f"  mean_loss_wrong   : {corr['mean_loss_wrong']:.4f}")

    print()
    print("All knor.py sanity checks PASSED.")
    sys.exit(0)
