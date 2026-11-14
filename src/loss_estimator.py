"""
loss_estimator.py
=================
Reconstruction-based per-patch loss estimator for LADP.

Pipeline overview
-----------------
  E  (N, dv)  — pre-connector CLIP visual features (raw)
  T  (N, dt)  — post-connector LLM tokens / hidden states (raw)

  During training  : fit ReconstructionNet  gφ : dt → dv
  At inference     : L_i = dist(E_std_i,  gφ(T_std_i))
                     patches with HIGH L are hard → prioritise

Classes
-------
  ReconstructionNet   — 3-layer MLP (gφ)
  NormStats           — fitted normalisation statistics
  LossEstimator       — inference-time loss computation
  ReconstructionTrainer — trains gφ on cached (E, T) tensors
  PatchDataset        — Dataset over pre-cached .pt files
"""

from __future__ import annotations

import os
import glob
import hashlib
import json
import math
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  ReconstructionNet  (gφ)
# ---------------------------------------------------------------------------

class ReconstructionNet(nn.Module):
    """
    MLP that reconstructs pre-connector features E from post-connector tokens T.

    Operates entirely in *standardised* space — the caller (LossEstimator /
    ReconstructionTrainer) is responsible for standardising inputs before
    passing them in.

    Architectures
    -------------
    mlp1       : linear(d_in, d_out)                     — no hidden layer
    mlp2       : linear(d_in, hidden) → GELU → Dropout → linear(hidden, d_out)
    mlp3_small : linear(d_in, hidden) → GELU → Dropout →
                 linear(hidden, hidden) → GELU → Dropout →
                 linear(hidden, d_out)
                 LLaVA (4096→2048→2048→1024): 14,685,184 params.  This is the
                 FYP-1 pilot model.  ``"mlp3"`` is accepted as an alias.
    mlp3_paper : linear(d_in, d_in) → LayerNorm → GELU → Dropout →
                 linear(d_in, h) → LayerNorm → GELU → Dropout →
                 linear(h, d_out),   h = min(hidden, max(d_in, d_out))
                 Exactly ``EmbeddingTransformer(num_layers=3)`` from the Lost in
                 Embeddings code (lyan62/vlm-info-loss,
                 scripts/reconstruct_embeddings_parallel.py), including PyTorch's
                 default Linear initialisation.  LLaVA (4096→4096→2048→1024):
                 27,282,432 params (paper Table 1: "27M").

    Parameters
    ----------
    d_in      : int   input dimension  dt  (post-connector tokens)   default 4096
    d_out     : int   output dimension dv  (pre-connector features)  default 1024
    hidden_dim: int   width of hidden layers                         default 2048
    depth     : str   one of DEPTHS (or the alias "mlp3")            default "mlp3_small"
    dropout   : float dropout probability between hidden layers      default 0.1

    Forward
    -------
    T : (B, d_in)  standardised post-connector tokens
    → Ê : (B, d_out)  predicted pre-connector features (unconstrained)
    """

    DEPTHS = ("mlp1", "mlp2", "mlp3_small", "mlp3_paper")
    ALIASES = {"mlp3": "mlp3_small"}

    def __init__(
        self,
        d_in: int = 4096,
        d_out: int = 1024,
        hidden_dim: int = 2048,
        depth: str = "mlp3_small",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        depth = self.ALIASES.get(depth, depth)
        if depth not in self.DEPTHS:
            raise ValueError(
                f"depth must be one of {self.DEPTHS} (or alias 'mlp3'); got {depth!r}"
            )

        self.d_in = d_in
        self.d_out = d_out
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout_p = dropout

        layers: List[nn.Module] = []

        if depth == "mlp1":
            # linear only — no hidden, no activation
            layers.append(nn.Linear(d_in, d_out))

        elif depth == "mlp2":
            # one hidden layer
            layers += [
                nn.Linear(d_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_out),
            ]

        elif depth == "mlp3_small":
            # two hidden layers of width hidden_dim (FYP-1 pilot model)
            layers += [
                nn.Linear(d_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_out),
            ]
            # NOTE: no activation after final linear — we want unconstrained Ê

        else:  # mlp3_paper — mirrors the source code line by line
            h = min(hidden_dim, max(d_in, d_out))
            if (d_in, d_out) == (4096, 1024) and h != 2048:
                # Still allowed (width ablation), but it is no longer the 27.3M
                # source model.  Configs pin the size via loss.expected_params.
                logger.warning(
                    "mlp3_paper with hidden_dim=%d (middle width %d) is not the 27,282,432-"
                    "parameter source model (needs hidden_dim=2048).", hidden_dim, h)
            layers += [
                nn.Linear(d_in, d_in),
                nn.LayerNorm(d_in),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_in, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, d_out),
            ]

        self.net = nn.Sequential(*layers)
        # The source keeps PyTorch's default nn.Linear init for the LLaVA MLP;
        # the other depths keep the original LADP init (pilot reproducibility).
        if depth != "mlp3_paper":
            self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming uniform for linear layers, zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, T: Tensor) -> Tensor:
        """
        Parameters
        ----------
        T : Tensor of shape (B, d_in)
            Standardised post-connector tokens.

        Returns
        -------
        Ê : Tensor of shape (B, d_out)
            Predicted pre-connector features in standardised space.
        """
        return self.net(T)  # (B, d_out)

    # ------------------------------------------------------------------
    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Known sizes, used to validate configs (see build_recon_net_from_config).
COCO_TRAIN2017_IMAGES = 118_287
MLP3_PAPER_LLAVA_PARAMS = 27_282_432    # mlp3_paper, 4096 -> 4096 -> 2048 -> 1024
MLP3_SMALL_LLAVA_PARAMS = 14_685_184    # mlp3_small, 4096 -> 2048 -> 2048 -> 1024

_LOSS_KEYS = {"reconstructor", "hidden_dim", "dropout", "distance", "norm", "expected_params"}


def build_recon_net_from_config(config: dict, d_in: int = 4096, d_out: int = 1024) -> ReconstructionNet:
    """Build g_phi from ``config["loss"]`` with no silent fallbacks.

    * ``loss.reconstructor`` is required: a missing or misspelt key used to fall
      back to the pilot model (``mlp3`` = ``mlp3_small``) without any message.
    * ``loss.expected_params`` (optional) must equal the built model's
      trainable parameter count, so a config cannot silently build a model of
      the wrong size (e.g. ``mlp3_paper`` with ``hidden_dim != 2048``).
    """
    loss_cfg = config.get("loss") or {}
    if "reconstructor" not in loss_cfg:
        unknown = sorted(set(loss_cfg) - _LOSS_KEYS)
        hint = f" Unrecognised loss keys: {unknown} (misspelt?)." if unknown else ""
        raise ValueError(
            "loss.reconstructor is missing from the config. Set it explicitly "
            "(mlp3_paper = 27.3M source model, mlp3_small = 14.7M pilot)." + hint)
    for key, default in (("hidden_dim", 2048), ("dropout", 0.1)):
        if key not in loss_cfg:
            logger.warning("loss.%s not set; using default %s", key, default)
    net = ReconstructionNet(
        d_in=d_in, d_out=d_out,
        hidden_dim=int(loss_cfg.get("hidden_dim", 2048)),
        depth=str(loss_cfg["reconstructor"]),
        dropout=float(loss_cfg.get("dropout", 0.1)),
    )
    expected = loss_cfg.get("expected_params")
    if expected is not None and net.param_count() != int(expected):
        raise ValueError(
            f"loss.expected_params={int(expected):,} but {net.depth} with hidden_dim="
            f"{net.hidden_dim} has {net.param_count():,} trainable parameters.")
    return net


# ---------------------------------------------------------------------------
# 2.  NormStats
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """
    Normalisation statistics fitted on the training split.

    Attributes
    ----------
    E_mean : Tensor  shape (dv,)   per-dimension mean of raw E on train set
    E_std  : Tensor  shape (dv,)   per-dimension std  of raw E on train set
    T_mean : Tensor  shape (dt,)   per-dimension mean of raw T on train set
    T_std  : Tensor  shape (dt,)   per-dimension std  of raw T on train set
    L_p5   : float   5th  percentile of per-patch MSE loss on train set
    L_p95  : float   95th percentile of per-patch MSE loss on train set
    """

    E_mean: Tensor   # (dv,)
    E_std:  Tensor   # (dv,)
    T_mean: Tensor   # (dt,)
    T_std:  Tensor   # (dt,)
    L_p5:   float    # 5th  percentile of per-patch loss on train set
    L_p95:  float    # 95th percentile of per-patch loss on train set
    # Which g_phi these statistics belong to: {"model_sha256", "depth",
    # "hidden_dim", "param_count", "run_name"}.  None for files written before
    # this field existed (e.g. the FYP-1 pilot), which cannot be verified.
    meta:   Optional[dict] = None

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise to disk via torch.save (atomically: tmp file + rename)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            {
                "E_mean": self.E_mean,
                "E_std":  self.E_std,
                "T_mean": self.T_mean,
                "T_std":  self.T_std,
                "L_p5":   self.L_p5,
                "L_p95":  self.L_p95,
                **({"meta": self.meta} if self.meta is not None else {}),
            },
            tmp,
        )
        os.replace(tmp, path)
        logger.info("NormStats saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "NormStats":
        """Deserialise from disk via torch.load."""
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            E_mean=d["E_mean"],
            E_std=d["E_std"],
            T_mean=d["T_mean"],
            T_std=d["T_std"],
            L_p5=float(d["L_p5"]),
            L_p95=float(d["L_p95"]),
            meta=d.get("meta"),
        )


def apply_norm_calibration(norm_stats: NormStats, calibration_path: Optional[str]) -> NormStats:
    """Return ``norm_stats`` with p5/p95 replaced from a calibration JSON (or unchanged).

    The JSON (e.g. produced by a loss-normalisation analysis) has the form
    ``{"L_p5": float, "L_p95": float, "source": str, "model_sha256": str, ...}``.
    If it records ``model_sha256``, it must match the stats' own fingerprint, so
    percentiles measured with one g_phi are never applied to another.
    Mean/std (the standardisation) are never changed — only the robust range.
    """
    if not calibration_path:
        return norm_stats
    with open(calibration_path, encoding="utf-8") as fh:
        cal = json.load(fh)
    p5, p95 = float(cal["L_p5"]), float(cal["L_p95"])
    if not (math.isfinite(p5) and math.isfinite(p95) and p95 > p5):
        raise ValueError(f"Invalid calibration in {calibration_path}: p5={p5}, p95={p95}")
    want = cal.get("model_sha256")
    have = (norm_stats.meta or {}).get("model_sha256")
    if want and have and want != have:
        raise ValueError(f"Calibration {calibration_path} was measured with a different g_phi "
                         f"({want[:12]}… vs {have[:12]}…).")
    meta = dict(norm_stats.meta or {})
    meta["percentiles_source"] = cal.get("source", calibration_path)
    logger.info("Loss percentiles from %s: p5=%.6f p95=%.6f (were %.6f / %.6f)",
                cal.get("source", calibration_path), p5, p95, norm_stats.L_p5, norm_stats.L_p95)
    return NormStats(norm_stats.E_mean, norm_stats.E_std, norm_stats.T_mean, norm_stats.T_std,
                     p5, p95, meta)


def state_dict_fingerprint(state_dict: dict) -> str:
    """SHA-256 over parameter names, shapes, dtypes and values (device-independent)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(f"{k}|{tuple(t.shape)}|{t.dtype}\n".encode("utf-8"))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def norm_stats_meta(net: "ReconstructionNet", run_name: Optional[str] = None) -> dict:
    return {"model_sha256": state_dict_fingerprint(net.state_dict()),
            "depth": getattr(net, "depth", None), "hidden_dim": getattr(net, "hidden_dim", None),
            "param_count": sum(p.numel() for p in net.parameters() if p.requires_grad),
            "run_name": run_name}


def check_norm_stats_pairing(norm_stats: NormStats, net: "ReconstructionNet",
                             require_meta: bool = False) -> None:
    """Refuse a NormStats that was fitted for a different g_phi (audit M4).

    * Stats with ``meta``: the weights' fingerprint, depth and parameter count
      must match ``net`` — otherwise ``ValueError``.
    * Legacy stats without ``meta`` (e.g. the FYP-1 pilot): cannot be verified;
      a warning is logged, or ``ValueError`` if ``require_meta``.
    * In both cases the stats' dimensions must match the network.
    """
    if (norm_stats.E_mean.numel(), norm_stats.T_mean.numel()) != (net.d_out, net.d_in):
        raise ValueError(f"NormStats dims (E {norm_stats.E_mean.numel()}, T {norm_stats.T_mean.numel()}) "
                         f"do not match g_phi (d_out {net.d_out}, d_in {net.d_in}).")
    m = norm_stats.meta
    if not m:
        msg = ("NormStats has no fingerprint (written before audit fix M4): cannot verify that "
               "it belongs to this checkpoint. Make sure both files come from the same run folder.")
        if require_meta:
            raise ValueError(msg)
        logger.warning(msg)
        return
    expected = norm_stats_meta(net)
    bad = [(k, m.get(k), expected[k]) for k in ("depth", "param_count", "model_sha256")
           if m.get(k) is not None and m.get(k) != expected[k]]
    if bad:
        raise ValueError("NormStats does not belong to this g_phi checkpoint: " + "; ".join(
            f"{k}: norm_stats={a!r} checkpoint={b!r}" for k, a, b in bad)
            + ". Use the norm_stats.pt written by the same training run as best.pt.")
    logger.info("NormStats verified against the checkpoint (sha256 %s…)", expected["model_sha256"][:12])


# ---------------------------------------------------------------------------
# 3.  LossEstimator
# ---------------------------------------------------------------------------

class LossEstimator:
    """
    Computes per-patch reconstruction loss at inference time.

    The estimator standardises raw (E, T) tensors using fitted NormStats,
    runs gφ, and measures the distance between the true and reconstructed
    E representations.

    Parameters
    ----------
    recon_net  : ReconstructionNet  trained gφ  (kept on its current device)
    norm_stats : NormStats          fitted statistics from training split
    distance   : str                one of {"mse", "norm_relative", "cosine"}
    """

    _DISTANCES = {"mse", "norm_relative", "cosine"}

    def __init__(
        self,
        recon_net: ReconstructionNet,
        norm_stats: NormStats,
        distance: str = "mse",
    ) -> None:
        if distance not in self._DISTANCES:
            raise ValueError(
                f"distance must be one of {self._DISTANCES}; got {distance!r}"
            )
        self.recon_net = recon_net
        self.norm_stats = norm_stats
        self.distance = distance

        # Infer device from recon_net
        self._device: torch.device = next(recon_net.parameters()).device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute(self, E: Tensor, T: Tensor) -> Tensor:
        """
        Compute per-patch raw reconstruction loss.

        Parameters
        ----------
        E : Tensor of shape (N, dv)
            Pre-connector features — raw (not standardised).
        T : Tensor of shape (N, dt)
            Post-connector tokens — raw (not standardised).

        Returns
        -------
        L : Tensor of shape (N,)
            Per-patch reconstruction loss (raw, not normalised to [0,1]).

        Steps
        -----
        1. Standardise E and T using NormStats.
        2. Predict Ê = gφ(T_std)  in standardised E-space.
        3. Compute distance(E_std, Ê) per patch.
        """
        ns = self.norm_stats
        dev = self._device

        # Move stats to the same device as the inputs (lazy)
        E_mean = ns.E_mean.to(dev)
        E_std_v = ns.E_std.to(dev)
        T_mean = ns.T_mean.to(dev)
        T_std_v = ns.T_std.to(dev)

        E = E.to(dev)
        T = T.to(dev)

        # 1. Standardise  --------------------------------------------------
        eps = 1e-8
        E_std = (E - E_mean) / (E_std_v + eps)   # (N, dv)
        T_std = (T - T_mean) / (T_std_v + eps)   # (N, dt)

        # 2. Reconstruct  --------------------------------------------------
        self.recon_net.eval()
        E_hat = self.recon_net(T_std)             # (N, dv)

        # 3. Distance  -----------------------------------------------------
        diff = E_std - E_hat                      # (N, dv)

        if self.distance == "mse":
            # L_i = ||E_std_i - Ê_i||²  (mean over dimensions)
            L = (diff ** 2).mean(dim=-1)          # (N,)

        elif self.distance == "norm_relative":
            # L_i = ||E_std_i - Ê_i||² / ||E_std_i||²
            num = (diff ** 2).sum(dim=-1)         # (N,)
            den = (E_std ** 2).sum(dim=-1) + eps  # (N,)
            L = num / den                         # (N,)

        else:  # cosine
            # L_i = 1 − cos_sim(E_std_i, Ê_i)
            cos = F.cosine_similarity(E_std, E_hat, dim=-1, eps=eps)  # (N,)
            L = 1.0 - cos                                              # (N,)

        return L  # (N,)

    # ------------------------------------------------------------------
    def compute_normalised(
        self,
        E: Tensor,
        T: Tensor,
        norm_mode: str = "robust",
    ) -> Tensor:
        """
        Compute per-patch loss normalised to [0, 1].

        Parameters
        ----------
        E        : Tensor of shape (N, dv)  raw pre-connector features
        T        : Tensor of shape (N, dt)  raw post-connector tokens
        norm_mode: str   "robust" or "per_image"

        Returns
        -------
        L_bar : Tensor of shape (N,)  clipped to [0, 1]

        Modes
        -----
        robust    : uses global p5/p95 from NormStats (fitted on train set)
                    L̄ = clip((L − p5) / (p95 − p5 + ε), 0, 1)
        per_image : min-max normalisation within this image's patches
                    L̄ = (L − L.min()) / (L.max() − L.min() + ε)
        """
        if norm_mode not in {"robust", "per_image"}:
            raise ValueError(
                f"norm_mode must be 'robust' or 'per_image'; got {norm_mode!r}"
            )

        L = self.compute(E, T)   # (N,)
        eps = 1e-8

        if norm_mode == "robust":
            p5  = L.new_tensor(self.norm_stats.L_p5)
            p95 = L.new_tensor(self.norm_stats.L_p95)
            L_bar = (L - p5) / (p95 - p5 + eps)
            L_bar = L_bar.clamp(0.0, 1.0)

        else:  # per_image
            L_min = L.min()
            L_max = L.max()
            L_bar = (L - L_min) / (L_max - L_min + eps)

        return L_bar  # (N,)


# ---------------------------------------------------------------------------
# 4.  PatchDataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """
    Dataset over pre-cached (E, T) tensors stored as .pt files.

    Expected file format
    --------------------
    cache_dir/features_{i}.pt  for i = 0, 1, 2, …
    Each file is a dict with keys:
        "E" : Tensor of shape (N_i, dv)   pre-connector features
        "T" : Tensor of shape (N_i, dt)   post-connector tokens

    This dataset flattens all images into individual *patch* items so that
    DataLoader with batch_size > 1 can be used efficiently.

    Parameters
    ----------
    cache_dir : str  directory containing features_{i}.pt files

    __getitem__ returns
    -------------------
    e : Tensor of shape (dv,)
    t : Tensor of shape (dt,)
    """

    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        self.cache_dir = cache_dir

        # Collect all .pt files matching the naming convention
        pattern = os.path.join(cache_dir, "features_*.pt")
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            raise FileNotFoundError(
                f"No 'features_*.pt' files found in {cache_dir!r}"
            )

        # Build an index: list of (file_path, patch_offset_within_file)
        # We also store cumulative counts for O(log N) __getitem__ lookups.
        self._file_paths: List[str] = file_paths
        self._cum_counts: List[int] = []   # cumulative patch counts

        total = 0
        for fp in file_paths:
            data = torch.load(fp, map_location="cpu", weights_only=True)
            n = data["E"].shape[0]
            total += n
            self._cum_counts.append(total)

        self._total = total
        logger.info(
            "PatchDataset: %d files, %d patches total", len(file_paths), total
        )

        # Cache for the most recently accessed file (avoid repeated I/O)
        self._cached_idx: Optional[int] = None
        self._cached_E:   Optional[Tensor] = None
        self._cached_T:   Optional[Tensor] = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        e : Tensor of shape (dv,)
        t : Tensor of shape (dt,)
        """
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range [0, {self._total})")

        # Binary search for which file this patch belongs to
        lo, hi = 0, len(self._cum_counts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum_counts[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        file_idx = lo

        # Offset within that file
        prev = self._cum_counts[file_idx - 1] if file_idx > 0 else 0
        patch_idx = idx - prev

        # Load file (use simple in-memory cache for single last file)
        if self._cached_idx != file_idx:
            data = torch.load(
                self._file_paths[file_idx], map_location="cpu", weights_only=True
            )
            self._cached_idx = file_idx
            self._cached_E = data["E"]  # (N_i, dv)
            self._cached_T = data["T"]  # (N_i, dt)

        e = self._cached_E[patch_idx]   # (dv,)
        t = self._cached_T[patch_idx]   # (dt,)
        return e, t


# ---------------------------------------------------------------------------
# 4b. ShardedPatchLoader  (fast, memory-bounded iteration for training)
# ---------------------------------------------------------------------------

class ShardedPatchLoader:
    """
    Iterate over cached feature files in *shards* of whole files.

    PatchDataset + DataLoader(shuffle=True) re-reads a ~6 MB file for almost
    every patch, which is unusable at COCO scale.  This loader instead:

      1. (optionally) shuffles the *file* order each epoch,
      2. loads ``files_per_shard`` files at once (prefetching the next shard
         in a background thread so disk I/O overlaps GPU compute),
      3. concatenates them, moves the shard to ``device``, shuffles the
         patches inside the shard and yields mini-batches.

    Memory use is bounded by ~2 shards regardless of dataset size.

    Yields
    ------
    e : Tensor (B, dv)   on ``device``, dtype as stored (usually fp16)
    t : Tensor (B, dt)
    """

    def __init__(
        self,
        file_paths: List[str],
        batch_size: int,
        files_per_shard: int = 64,
        shuffle: bool = True,
        seed: int = 42,
        device: "torch.device | str" = "cpu",
    ) -> None:
        if not file_paths:
            raise ValueError("ShardedPatchLoader needs at least one file")
        self.file_paths = list(file_paths)
        self.batch_size = int(batch_size)
        self.files_per_shard = max(1, int(files_per_shard))
        self.shuffle = shuffle
        self.seed = int(seed)
        self.device = torch.device(device)
        self._epoch = 0

    def num_batches(self, patches_per_file: int) -> int:
        """Exact batches per epoch when every file holds ``patches_per_file``.

        Batches never cross shard boundaries, so this is the sum over shards
        of ceil(shard_patches / batch_size), not ceil(total / batch_size).
        """
        n = len(self.file_paths)
        full, rest = divmod(n, self.files_per_shard)
        per_full = math.ceil(self.files_per_shard * patches_per_file / self.batch_size)
        return full * per_full + (math.ceil(rest * patches_per_file / self.batch_size) if rest else 0)

    @staticmethod
    def _load_shard(paths: List[str]) -> Tuple[Tensor, Tensor]:
        Es, Ts = [], []
        for fp in paths:
            d = torch.load(fp, map_location="cpu", weights_only=True)
            Es.append(d["E"])
            Ts.append(d["T"])
        return torch.cat(Es, 0), torch.cat(Ts, 0)

    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor

        gen = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1

        order = list(range(len(self.file_paths)))
        if self.shuffle:
            order = torch.randperm(len(order), generator=gen).tolist()
        paths = [self.file_paths[i] for i in order]
        shards = [
            paths[i : i + self.files_per_shard]
            for i in range(0, len(paths), self.files_per_shard)
        ]

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._load_shard, shards[0])
            for s in range(len(shards)):
                E, T = fut.result()
                if s + 1 < len(shards):
                    fut = pool.submit(self._load_shard, shards[s + 1])
                E = E.to(self.device, non_blocking=True)
                T = T.to(self.device, non_blocking=True)
                n = E.shape[0]
                if self.shuffle:
                    perm = torch.randperm(n, generator=gen).to(self.device)
                else:
                    perm = torch.arange(n, device=self.device)
                for b in range(0, n, self.batch_size):
                    idx = perm[b : b + self.batch_size]
                    yield E[idx], T[idx]
                del E, T


# ---------------------------------------------------------------------------
# 5.  ReconstructionTrainer
# ---------------------------------------------------------------------------

class ReconstructionTrainer:
    """
    Trains gφ (ReconstructionNet) on cached (E, T) tensors.

    The trainer:
      1. Loads PatchDataset from cache_dir.
      2. Splits into train / val by config["training"]["val_fraction"].
      3. Computes dataset-level mean / std for E and T (standardisation).
      4. Trains with Adam, early stopping on val MSE.
      5. Computes p5 / p95 of per-patch train loss for robust normalisation.
      6. Saves checkpoint + NormStats to config["checkpoints_dir"]/recon/.

    Parameters
    ----------
    recon_net : ReconstructionNet   the model to train (modified in-place)
    config    : dict                LADP config (see configs/base_*.yaml)
    device    : str                 "cuda" | "cpu"
    """

    def __init__(
        self,
        recon_net: ReconstructionNet,
        config: dict,
        device: str = "cpu",
    ) -> None:
        self.recon_net = recon_net
        self.config = config
        self.device = torch.device(device)
        self.recon_net.to(self.device)

        training_cfg = config.get("training", {})
        # Defaults below reproduce the original (pilot) trainer.  Say which
        # settings fall back to them, so a missing key is never silent.
        core = ["lr", "batch_size", "epochs", "patience", "val_fraction", "seed",
                "files_per_shard", "optimizer", "scheduler", "grad_clip",
                "loss_reduction", "min_delta"]
        if str(training_cfg.get("scheduler", "none")).lower() != "none":
            core += ["min_lr_ratio"] + ([] if "warmup_ratio" in training_cfg else ["warmup_steps"])
        missing = [k for k in core if k not in training_cfg]
        if missing:
            logger.warning("training.%s not set in the config; using the trainer defaults "
                           "(original LADP / pilot behaviour).", ", training.".join(missing))
        self.lr          = float(training_cfg.get("lr", 1e-4))
        self.batch_size  = int(training_cfg.get("batch_size", 128))
        self.epochs      = int(training_cfg.get("epochs", 30))
        self.patience    = int(training_cfg.get("patience", 5))
        self.val_frac    = float(training_cfg.get("val_fraction", 0.05))
        self.num_workers = int(training_cfg.get("num_workers", 4))
        self.seed        = int(training_cfg.get("seed", 42))

        # Optimisation options.  Defaults reproduce the original LADP trainer
        # (Adam, constant lr, no clipping, min_delta 0) so old configs and the
        # FYP-1 pilot run behave exactly as before.
        self.optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
        if self.optimizer_name not in ("adam", "adamw"):
            raise ValueError(f"training.optimizer must be 'adam' or 'adamw'; got {self.optimizer_name!r}")
        wd = training_cfg.get("weight_decay")
        self.weight_decay = None if wd is None else float(wd)  # None → torch default
        self.scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
        if self.scheduler_name not in ("none", "warmup_cosine"):
            raise ValueError(
                f"training.scheduler must be 'none' or 'warmup_cosine'; got {self.scheduler_name!r}"
            )
        self.warmup_steps = int(training_cfg.get("warmup_steps", 2000))
        self._warmup_steps_cfg = self.warmup_steps   # as configured (resume check, R1)
        wr = training_cfg.get("warmup_ratio")   # if set, overrides warmup_steps
        self.warmup_ratio = None if wr is None else float(wr)
        if self.warmup_ratio is not None and not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"training.warmup_ratio must be in [0, 1); got {self.warmup_ratio}")
        self.min_lr_ratio = float(training_cfg.get("min_lr_ratio", 0.01))
        gc = training_cfg.get("grad_clip")
        self.grad_clip = None if gc in (None, 0, "none") else float(gc)
        self.min_delta = float(training_cfg.get("min_delta", 0.0))
        self.loss_reduction = str(training_cfg.get("loss_reduction", "sum")).lower()
        if self.loss_reduction not in ("sum", "mean"):
            raise ValueError(f"training.loss_reduction must be 'sum' or 'mean'; got {self.loss_reduction!r}")

        # Number of cached images to use (train + val); None = all available.
        mi = config.get("data", {}).get("max_images")
        self.max_images = None if mi in (None, "all") else int(mi)
        self.subset_mode = str(config.get("data", {}).get("subset_mode", "random")).lower()
        if self.subset_mode not in ("random", "first"):
            raise ValueError(f"data.subset_mode must be 'random' or 'first'; got {self.subset_mode!r}")
        # Refuse to train on a partially built cache (full-COCO run).
        self.require_complete_cache = bool(config.get("data", {}).get("require_complete_cache", False))
        # Number of images a "use everything" run (max_images: null) must see,
        # e.g. 118287 for COCO train2017.  Required with require_complete_cache.
        ei = config.get("data", {}).get("expected_images")
        self.expected_images = None if ei is None else int(ei)
        # Resume refuses a checkpoint from a different split / batch / model /
        # optimiser setup unless this is set.
        self.allow_resume_mismatch = bool(training_cfg.get("allow_resume_mismatch", False))
        # JSON file of validation images that must stay in val (e.g. the
        # pilot's 500), also evaluated separately as "val_ref" each epoch.
        self.fixed_val_images = training_cfg.get("fixed_val_images")

        self._scheduler = None
        self._last_epoch_steps = 0
        self.train_loss_history: List[float] = []
        self.val_loss_history:   List[float] = []
        self.val_ref_loss_history: List[float] = []

        self.ckpt_dir = os.path.join(
            config.get("checkpoints_dir", "checkpoints"),
            str(training_cfg.get("run_name", "recon")),
        )

    # ------------------------------------------------------------------
    def _make_optimizer(self) -> torch.optim.Optimizer:
        kw = {"lr": self.lr}
        if self.weight_decay is not None:
            kw["weight_decay"] = self.weight_decay
        cls = torch.optim.AdamW if self.optimizer_name == "adamw" else torch.optim.Adam
        return cls(self.recon_net.parameters(), **kw)

    # ------------------------------------------------------------------
    @staticmethod
    def _lr_factor(schedule: dict, floor: float, step: int) -> float:
        """lr / base_lr at optimiser step ``step`` for a schedule dict.

        schedule = {"warmup_steps": W, "total_steps": T,
                    "rebase_step": s0 | None, "rebase_factor": f0 | None}
          step < W                  : step / W                       (linear warm-up)
          step >= s0 (if rebased)   : cosine from f0 down to floor over [s0, T]
          otherwise                 : cosine from 1 down to floor over [W, T]
        Progress is clamped at 1, so steps past T stay at the floor.
        A rebase is created when a run is resumed with more (or fewer) epochs:
        the lr continues from its current value instead of jumping (R1).
        """
        W, T = int(schedule["warmup_steps"]), int(schedule["total_steps"])
        if W > 0 and step < W:
            return step / W
        s0, f0 = schedule.get("rebase_step"), schedule.get("rebase_factor")
        if s0 is not None and step >= s0:
            p = min(1.0, (step - s0) / max(1, T - s0))
            return floor + (float(f0) - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
        p = min(1.0, (step - W) / max(1, T - W))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

    def _resume_schedule(self, ckpt: dict, total_steps: int, current: dict) -> dict:
        """Schedule that continues a checkpoint's lr curve under a new ``total_steps`` (R1).

        * Warm-up length is frozen at the checkpoint's value (those steps are
          already done; with warmup_ratio it would otherwise be re-derived from
          the new total).
        * Same total -> the checkpoint's schedule unchanged (exact resume).
        * Different total -> if the resume point is past warm-up, the cosine is
          re-based: it starts at the checkpoint's current factor and reaches
          the floor at the new total, so lr is continuous at the resume point.
        """
        ri = ckpt.get("run_info") or {}
        saved = ckpt.get("schedule") or (
            {"warmup_steps": ri["warmup_steps"], "total_steps": ri["total_steps"],
             "rebase_step": None, "rebase_factor": None}
            if "warmup_steps" in ri and "total_steps" in ri else None)
        if saved is None:                       # pre-R1 checkpoint without schedule info
            return current
        done = int(ckpt["scheduler"]["last_epoch"])
        if int(saved["total_steps"]) == int(total_steps):
            return dict(saved)
        if total_steps < done:
            raise ValueError(f"Cannot resume with {total_steps} total steps: the checkpoint has already "
                             f"taken {done}. Increase training.epochs.")
        W = int(saved["warmup_steps"])
        new = {"warmup_steps": W, "total_steps": int(total_steps), "rebase_step": None, "rebase_factor": None}
        if done > W:
            new["rebase_step"] = done
            new["rebase_factor"] = self._lr_factor(saved, self.min_lr_ratio, done)
        logger.info("Resuming with a new schedule length: %d -> %d total steps; warm-up kept at %d steps; "
                    "lr continues from %.3e at step %d.", int(saved["total_steps"]), int(total_steps), W,
                    self.lr * self._lr_factor(new, self.min_lr_ratio, done), done)
        return new

    def _make_scheduler(self, optimiser, total_steps: int, last_step: int = 0,
                        schedule: Optional[dict] = None):
        """Linear warm-up then cosine decay to ``min_lr_ratio``·lr, per step.

        Same curve as the source's SequentialLR(LambdaLR(min(1, s/warmup)),
        CosineAnnealingLR(T_max=total-warmup, eta_min=0.01·lr)).  Progress is
        clamped at 1, so overshooting ``total_steps`` keeps lr at the minimum.

        ``last_step`` > 0 starts the schedule as if ``last_step`` optimiser
        steps had already been taken (resume without saved scheduler state).
        ``schedule`` (see ``_lr_factor``) overrides warm-up/total, e.g. after
        an epoch extension.
        """
        if self.scheduler_name == "none":
            return None
        sched = dict(schedule) if schedule is not None else {
            "warmup_steps": max(0, self.warmup_steps), "total_steps": int(total_steps),
            "rebase_step": None, "rebase_factor": None}
        floor = self.min_lr_ratio

        def lr_lambda(step: int) -> float:
            return self._lr_factor(sched, floor, step)

        if last_step <= 0:
            return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
        for g in optimiser.param_groups:
            g.setdefault("initial_lr", self.lr)
        # last_epoch = k-1: the constructor's initial step moves it to k and
        # sets lr = base_lr * lr_lambda(k), exactly as after k real steps.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda, last_epoch=last_step - 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _check_cache_complete(cache_dir: str, n_files: int,
                              required_images: Optional[int] = None) -> int:
        """Refuse a partial cache.

        * the manifest must exist and carry ``"complete": true`` — manifests
          written before that key existed (e.g. the 10k pilot cache) are
          rejected rather than assumed complete;
        * ``cached_ok`` must equal ``total_images``;
        * with ``required_images`` (the number of images this run needs), the
          manifest and the files on disk must cover at least that many.
        """
        mpath = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise RuntimeError(f"data.require_complete_cache is set but {mpath} is missing; "
                               "run cache_features.py to completion first.")
        with open(mpath, encoding="utf-8") as fh:
            m = json.load(fh)
        total, ok = int(m.get("total_images", -1)), int(m.get("cached_ok", -2))
        if "complete" not in m:
            raise RuntimeError(
                f"Feature cache manifest {mpath} has no 'complete' flag (older cache_features.py "
                f"format; total_images={total}, cached_ok={ok}). It cannot be trusted as a complete "
                "cache. Re-run cache_features.py for this split: it re-verifies existing files, "
                "extracts any missing ones and writes a manifest with 'complete'.")
        # Images excluded explicitly by cache_features.py (--exclude_failed),
        # each recorded with its error; they count as accounted for, not as cached.
        n_excl = len(m.get("excluded", []))
        if m["complete"] is not True or ok + n_excl != total or n_files < ok:
            raise RuntimeError(
                f"Feature cache is incomplete: manifest total_images={total}, cached_ok={ok}, "
                f"excluded={n_excl}, files on disk={n_files}, complete={m.get('complete')}. "
                "Finish cache_features.py (it resumes) before training with "
                "data.require_complete_cache: true.")
        if required_images is not None and (total < required_images or n_files < required_images - n_excl):
            raise RuntimeError(
                f"Feature cache is complete for {total} images ({n_files} files on disk), but this "
                f"run needs {required_images}. Build the cache for all {required_images} images "
                "first (cache_features.py).")
        if n_excl:
            logger.warning("Cache complete with %d explicitly excluded image(s) (see manifest 'excluded'): "
                           "%d of %d images are used.", n_excl, ok, total)
        logger.info("Cache complete: %d images (manifest verified)", total)
        return n_excl

    # ------------------------------------------------------------------
    def _load_fixed_val(self, cache_dir: str, file_paths: List[str]) -> List[str]:
        """Resolve training.fixed_val_images to cache file paths (or [])."""
        if not self.fixed_val_images:
            return []
        with open(self.fixed_val_images, encoding="utf-8") as fh:
            spec = json.load(fh)
        names = list(spec["files"])
        by_name = {os.path.basename(p): p for p in file_paths}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"{len(missing)} fixed validation file(s) not in the cache, "
                             f"e.g. {missing[:3]} (from {self.fixed_val_images})")
        # Check the files still hold the same images (by image file name).
        images = spec.get("images")
        mpath = os.path.join(cache_dir, "manifest.json")
        if images and os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as fh:
                fmap = {r["filename"]: os.path.basename(r["image_path"])
                        for r in json.load(fh).get("files", [])}
            bad = [(n, im) for n, im in zip(names, images) if n in fmap and fmap[n] != im]
            if bad:
                raise ValueError(f"Fixed validation files now hold different images, e.g. {bad[:3]}")
        logger.info("Fixed validation set: %d images from %s", len(names), self.fixed_val_images)
        return sorted(by_name[n] for n in names)

    # ------------------------------------------------------------------
    @staticmethod
    def _atomic_save(obj, path: str) -> None:
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    @staticmethod
    def _fingerprint_names(*lists) -> str:
        """Hash of file *names* (order kept) — identifies a split."""
        h = hashlib.sha256()
        for lst in lists:
            for p in lst:
                h.update(os.path.basename(p).encode("utf-8") + b"\0")
        return h.hexdigest()

    @staticmethod
    def _file_fingerprint(path: str) -> str:
        """Weights fingerprint of a state-dict file (same as state_dict_fingerprint)."""
        return state_dict_fingerprint(torch.load(path, map_location="cpu", weights_only=True))

    @staticmethod
    def _fingerprint_files(paths: List[str]) -> str:
        """Hash of name, size and mtime of each file — changes if a file is rebuilt."""
        h = hashlib.sha256()
        for p in paths:
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode("utf-8"))
        return h.hexdigest()

    def _train_stats_dict(self, n_train: int, files_fp: str) -> dict:
        return {"E_mean": self._E_mean, "E_std": self._E_std_v,
                "T_mean": self._T_mean, "T_std": self._T_std_v,
                "n_train": int(n_train), "files_fp": files_fp}

    def _find_train_stats(self, ckpt: Optional[dict], files_fp: str):
        """Saved mean/std for exactly these train files, from last.pt or train_stats.pt."""
        candidates = [("checkpoint", ckpt.get("train_stats") if ckpt else None)]
        p = os.path.join(self.ckpt_dir, "train_stats.pt")
        if os.path.exists(p):
            try:
                candidates.append((p, torch.load(p, map_location="cpu", weights_only=True)))
            except Exception as exc:          # corrupt file: recompute
                logger.warning("Could not read %s (%s); recomputing statistics.", p, exc)
        for src, s in candidates:
            if s and s.get("files_fp") == files_fp:
                logger.info("Normalisation statistics taken from %s", src)
                return (s["E_mean"].float().cpu(), s["E_std"].float().cpu(),
                        s["T_mean"].float().cpu(), s["T_std"].float().cpu(), int(s["n_train"]))
            if s:
                logger.info("Saved statistics in %s are for different train files; ignoring.", src)
        return None

    _RESUME_STRICT = ("split_fp", "n_train_files", "n_val_files", "n_train_patches", "batch_size",
                      "files_per_shard", "steps_per_epoch", "depth", "param_count", "optimizer",
                      "weight_decay", "lr", "scheduler", "warmup_config", "min_lr_ratio",
                      "grad_clip", "loss_reduction")
    _RESUME_SOFT = ("epochs", "total_steps", "patience", "min_delta")

    def _check_resume_compat(self, saved: Optional[dict], current: dict) -> None:
        """Refuse to continue a checkpoint under a different split/model/optimiser setup."""
        if not saved:
            logger.warning("Checkpoint has no run_info (older format): cannot check that the split, "
                           "batch size, model and optimiser match. Use the config it was trained with.")
            return
        diff = lambda keys: [(k, saved[k], current.get(k)) for k in keys
                             if k in saved and saved[k] != current.get(k)]
        for k, old, new in diff(self._RESUME_SOFT):
            logger.warning("Resume: %s changed %s -> %s (the run will differ from an "
                           "uninterrupted one).", k, old, new)
        bad = diff(self._RESUME_STRICT)
        if bad:
            txt = "; ".join(f"{k}: checkpoint={old!r} now={new!r}" for k, old, new in bad)
            if self.allow_resume_mismatch:
                logger.warning("Resume mismatch allowed by training.allow_resume_mismatch: %s", txt)
            else:
                raise ValueError("Checkpoint does not match this run (" + txt + "). Resume with the "
                                 "original config and cache, or set training.allow_resume_mismatch: "
                                 "true to override.")

    # ------------------------------------------------------------------
    def fit(
        self,
        cache_dir: str,
        resume_from: Optional[str] = None,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
    ) -> Tuple[ReconstructionNet, NormStats]:
        """
        Train gφ and return the best model + fitted NormStats.

        Parameters
        ----------
        cache_dir     : str  directory with features_{i}.pt files
        resume_from   : str  optional checkpoint to resume from.  Either a full
                             resume checkpoint (``last.pt``: model, optimiser,
                             epoch, best_val_loss, patience_count) or a bare
                             state_dict (``best.pt``), in which case
                             ``start_epoch`` / ``best_val_loss`` must be given
                             and a fresh optimiser is used.
        start_epoch   : int  first epoch to run (ignored for full checkpoints)
        best_val_loss : float best val loss so far (ignored for full checkpoints)

        Returns
        -------
        recon_net  : ReconstructionNet  best checkpoint (loaded in-place)
        norm_stats : NormStats          fitted statistics
        """
        torch.manual_seed(self.seed)

        # 1. Image-level train/val split  ----------------------------------
        # Split by *file* (= image) so patches of one image never appear in
        # both train and val (a patch-level split leaks near-duplicates).
        file_paths = sorted(glob.glob(os.path.join(cache_dir, "features_*.pt")))
        if not file_paths:
            raise FileNotFoundError(
                f"No 'features_*.pt' files found in {cache_dir!r}"
            )
        # How many images this run needs: max_images, or (max_images: null =
        # "all") the declared full dataset size.  Without the latter a partial
        # cache (e.g. the 10k pilot cache) would pass as the full one.
        if self.max_images is None:
            required = self.expected_images
            if self.require_complete_cache and required is None:
                raise ValueError(
                    "data.require_complete_cache is set with data.max_images: null (use all images) "
                    "but data.expected_images is not set. Set it to the full dataset size "
                    f"({COCO_TRAIN2017_IMAGES} for COCO train2017) so a partial cache cannot be "
                    "mistaken for the full one.")
        else:
            required = self.max_images
            if self.max_images > len(file_paths):
                logger.warning("data.max_images=%d but only %d cached images exist; using all %d.",
                               self.max_images, len(file_paths), len(file_paths))
        n_excluded = 0
        if self.require_complete_cache:
            n_excluded = self._check_cache_complete(cache_dir, len(file_paths), required)
        if self.max_images is None and self.expected_images is not None \
                and len(file_paths) != self.expected_images - n_excluded:
            raise RuntimeError(
                f"data.expected_images={self.expected_images} ({n_excluded} explicitly excluded) but "
                f"{len(file_paths)} feature files are in {cache_dir}. A max_images: null run must see "
                "the full cache.")
        fixed_val = self._load_fixed_val(cache_dir, file_paths)   # [] if not configured

        if (self.max_images is not None and self.max_images < len(file_paths)
                and self.subset_mode == "first"):
            # First N files = first N sorted images (the pilot's 10k cache).
            file_paths = sorted(file_paths)[: self.max_images]
            outside = sorted(set(fixed_val) - set(file_paths))
            if outside:
                raise ValueError(f"{len(outside)} fixed validation images are outside the first "
                                 f"{self.max_images} files, e.g. {outside[:3]}")
            logger.info("Using the first %d cached images (data.subset_mode=first)", self.max_images)
        elif self.max_images is not None and self.max_images < len(file_paths):
            # Deterministic random subset of images (seeded), taken *before*
            # the image-level split so val stays disjoint by image.  Fixed
            # validation images are always kept.
            fixed_set = set(fixed_val)
            rest = [p for p in file_paths if p not in fixed_set]
            k = max(0, self.max_images - len(fixed_val))
            sub = torch.randperm(
                len(rest), generator=torch.Generator().manual_seed(self.seed + 1)
            )[:k].tolist()
            logger.info("Using %d of %d cached images (data.max_images)",
                        self.max_images, len(file_paths))
            file_paths = sorted(fixed_val + [rest[i] for i in sub])
        if len(file_paths) < 2:
            raise ValueError("Need at least 2 cached images for a train/val split")
        n_val = min(len(file_paths) - 1, max(1, int(len(file_paths) * self.val_frac)))
        if fixed_val:
            # Pinned images (e.g. the pilot's 500 val images) stay in val; the
            # rest of val is drawn from the remaining images.
            fixed_set = set(fixed_val)
            rest = [p for p in file_paths if p not in fixed_set]
            n_extra = max(0, n_val - len(fixed_val))
            perm = torch.randperm(
                len(rest), generator=torch.Generator().manual_seed(self.seed)
            ).tolist()
            val_files   = sorted(fixed_val + [rest[i] for i in perm[:n_extra]])
            train_files = sorted(rest[i] for i in perm[n_extra:])
        else:
            perm = torch.randperm(
                len(file_paths), generator=torch.Generator().manual_seed(self.seed)
            ).tolist()
            val_files   = sorted(file_paths[i] for i in perm[:n_val])
            train_files = sorted(file_paths[i] for i in perm[n_val:])
        assert not set(train_files) & set(val_files), "train/val overlap"
        logger.info("Train images: %d  |  Val images: %d%s", len(train_files), len(val_files),
                    f"  (incl. {len(fixed_val)} fixed reference images)" if fixed_val else "")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(os.path.join(self.ckpt_dir, "split.json"), "w", encoding="utf-8") as fh:
            json.dump({"train": [os.path.basename(p) for p in train_files],
                       "val": [os.path.basename(p) for p in val_files],
                       "val_ref": [os.path.basename(p) for p in fixed_val]}, fh)

        files_per_shard = int(
            self.config.get("training", {}).get("files_per_shard", 64)
        )
        train_loader = ShardedPatchLoader(
            train_files, self.batch_size, files_per_shard,
            shuffle=True, seed=self.seed, device=self.device,
        )
        val_loader = ShardedPatchLoader(
            val_files, self.batch_size, files_per_shard,
            shuffle=False, seed=self.seed, device=self.device,
        )
        val_ref_loader = None
        if fixed_val and len(fixed_val) < len(val_files):
            val_ref_loader = ShardedPatchLoader(
                fixed_val, self.batch_size, files_per_shard,
                shuffle=False, seed=self.seed, device=self.device,
            )
        # Unshuffled pass over train files (stats / p5-p95 do not need shuffling)
        train_scan = ShardedPatchLoader(
            train_files, 8192, files_per_shard,
            shuffle=False, seed=self.seed, device=self.device,
        )

        # Resume checkpoint is read here so its saved stats can be reused.
        ckpt, ckpt_full = None, False
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            ckpt_full = isinstance(ckpt, dict) and "model" in ckpt and "optimizer" in ckpt
            if (os.path.normcase(os.path.abspath(os.path.dirname(resume_from)))
                    != os.path.normcase(os.path.abspath(self.ckpt_dir))):
                logger.warning("Resuming from %s, but this run writes checkpoints to %s "
                               "(training.run_name).", resume_from, self.ckpt_dir)

        # 2. Streaming standardisation stats from train split  -------------
        # Saved right after this pass (train_stats.pt) and in every last.pt,
        # and reused on resume when the train files are unchanged, so a resume
        # does not repeat a full pass over the cache.
        files_fp = self._fingerprint_files(train_files)
        stats = self._find_train_stats(ckpt if ckpt_full else None, files_fp) if resume_from else None
        if stats is not None:
            (self._E_mean, self._E_std_v, self._T_mean, self._T_std_v, n_train) = stats
            logger.info("Reusing saved normalisation statistics (train files unchanged)")
        else:
            logger.info("Computing normalisation statistics on train split (streaming)…")
            (self._E_mean, self._E_std_v,
             self._T_mean, self._T_std_v, n_train) = self._streaming_mean_std(train_scan)
            self._atomic_save(self._train_stats_dict(n_train, files_fp),
                              os.path.join(self.ckpt_dir, "train_stats.pt"))
        logger.info("Train patches: %d", n_train)

        # 3. Train  ---------------------------------------------------------
        optimiser = self._make_optimizer()
        # Batches never cross shard boundaries, so count them per shard.
        if n_train % len(train_files) == 0:
            ppf = n_train // len(train_files)
            steps_per_epoch = train_loader.num_batches(ppf)
            if (files_per_shard * ppf) % self.batch_size:
                logger.warning(
                    "files_per_shard x patches/file = %d is not a multiple of batch_size=%d: every "
                    "shard ends with a smaller batch (%d steps/epoch instead of %d).",
                    files_per_shard * ppf, self.batch_size, steps_per_epoch,
                    math.ceil(n_train / self.batch_size))
        else:
            steps_per_epoch = math.ceil(n_train / self.batch_size)
            logger.warning("Cached files hold different numbers of patches; steps/epoch (%d) is "
                           "an estimate and the lr schedule may end early or late.", steps_per_epoch)
        total_steps = self.epochs * steps_per_epoch
        if self.warmup_ratio is not None:
            self.warmup_steps = int(round(self.warmup_ratio * total_steps))
        schedule = {"warmup_steps": max(0, self.warmup_steps), "total_steps": int(total_steps),
                    "rebase_step": None, "rebase_factor": None}
        if ckpt_full and self.scheduler_name != "none" and ckpt.get("scheduler") is not None:
            # continue the checkpoint's lr curve (handles epoch extensions, R1)
            schedule = self._resume_schedule(ckpt, total_steps, schedule)
            self.warmup_steps = int(schedule["warmup_steps"])
        self._schedule = schedule
        self._scheduler = self._make_scheduler(optimiser, total_steps, schedule=schedule)
        logger.info(
            "Optimiser=%s lr=%.2e wd=%s | grad_clip=%s | patience=%d min_delta=%g",
            self.optimizer_name, self.lr,
            "default" if self.weight_decay is None else self.weight_decay,
            self.grad_clip, self.patience, self.min_delta,
        )
        logger.info(
            "Batch=%d patches (%.1f images) | steps/epoch=%d | total steps=%d (%d epochs) | "
            "scheduler=%s warmup=%d steps (%.1f%% of total) min_lr_ratio=%.3f",
            self.batch_size, self.batch_size / 576, steps_per_epoch, total_steps, self.epochs,
            self.scheduler_name, self.warmup_steps,
            100.0 * self.warmup_steps / max(1, total_steps), self.min_lr_ratio,
        )
        if self.scheduler_name != "none" and self.warmup_steps > 0.10 * total_steps:
            logger.warning("Warm-up is %.0f%% of all steps (> 10%%). For small runs set "
                           "training.warmup_ratio (e.g. 0.05-0.10) instead of warmup_steps.",
                           100.0 * self.warmup_steps / max(1, total_steps))

        patience_count = 0
        os.makedirs(self.ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(self.ckpt_dir, "best.pt")
        last_ckpt_path = os.path.join(self.ckpt_dir, "last.pt")
        rng_state = None
        run_info = {
            # must match on resume (else the resumed run is a different experiment)
            "split_fp": self._fingerprint_names(train_files, ["|"], val_files),
            "n_train_files": len(train_files), "n_val_files": len(val_files),
            "n_train_patches": int(n_train), "batch_size": self.batch_size,
            "files_per_shard": files_per_shard, "steps_per_epoch": int(steps_per_epoch),
            "depth": getattr(self.recon_net, "depth", type(self.recon_net).__name__),
            "param_count": sum(p.numel() for p in self.recon_net.parameters() if p.requires_grad),
            "optimizer": self.optimizer_name, "weight_decay": self.weight_decay, "lr": self.lr,
            "scheduler": self.scheduler_name,
            # the CONFIGURED warm-up must match; the derived step count may
            # differ when epochs change with warmup_ratio (R1)
            "warmup_config": (["ratio", self.warmup_ratio] if self.warmup_ratio is not None
                              else ["steps", int(self._warmup_steps_cfg)]),
            "warmup_steps": int(self.warmup_steps),           # effective (informational)
            "min_lr_ratio": self.min_lr_ratio, "grad_clip": self.grad_clip,
            "loss_reduction": self.loss_reduction,
            # may change (logged as a warning)
            "epochs": self.epochs, "total_steps": int(total_steps),
            "patience": self.patience, "min_delta": self.min_delta,
        }
        skip_training = False
        best_sha: Optional[str] = None    # fingerprint of the weights best.pt must hold (R3)

        if ckpt is not None:
            if ckpt_full:
                self._check_resume_compat(ckpt.get("run_info"), run_info)
                self.recon_net.load_state_dict(ckpt["model"])
                optimiser.load_state_dict(ckpt["optimizer"])
                start_epoch    = int(ckpt["epoch"]) + 1
                best_val_loss  = float(ckpt["best_val_loss"])
                patience_count = int(ckpt["patience_count"])
                ckpt_lr = optimiser.param_groups[0].get("initial_lr",
                                                        optimiser.param_groups[0]["lr"])
                if self._scheduler is None and abs(ckpt_lr - self.lr) > 1e-12:
                    logger.warning("Checkpoint lr %.2e differs from config lr %.2e; "
                                   "keeping the checkpoint's optimiser state.", ckpt_lr, self.lr)
                if self._scheduler is not None:
                    if ckpt.get("scheduler") is not None:
                        self._scheduler.load_state_dict(ckpt["scheduler"])
                    else:
                        done = (start_epoch - 1) * steps_per_epoch
                        self._scheduler = self._make_scheduler(optimiser, total_steps, last_step=done)
                        logger.warning("Checkpoint has no scheduler state: lr schedule fast-forwarded "
                                       "to step %d (lr=%.3e).", done, optimiser.param_groups[0]["lr"])
                if "history" in ckpt:
                    self.train_loss_history = list(ckpt["history"]["train"])
                    self.val_loss_history   = list(ckpt["history"]["val"])
                    self.val_ref_loss_history = list(ckpt["history"].get("val_ref", []))
                rng_state = ckpt.get("rng")
                if patience_count > 0 and patience_count >= self.patience:
                    skip_training = True
                    logger.info("Early stopping had already triggered in this checkpoint; "
                                "no further epochs, finalising only (p5/p95, norm_stats.pt).")
            else:
                self.recon_net.load_state_dict(ckpt)
                logger.warning(
                    "Resume checkpoint has weights only — optimiser state (Adam moments) starts "
                    "fresh and the run cannot be checked against its original settings."
                )
                if self._scheduler is not None and start_epoch > 1:
                    # Do not replay the warm-up: continue the schedule where
                    # epoch start_epoch-1 ended.
                    done = (start_epoch - 1) * steps_per_epoch
                    self._scheduler = self._make_scheduler(optimiser, total_steps, last_step=done)
                    logger.warning("lr schedule fast-forwarded to step %d of %d (lr=%.3e).",
                                   done, total_steps, optimiser.param_groups[0]["lr"])
            # best.pt must exist in this run's folder AND hold this run's best
            # weights, or the final reload would silently use other weights (R3).
            if ckpt_full:
                best_sha = ckpt.get("best_sha256")
                if not os.path.exists(best_ckpt_path):
                    if math.isfinite(best_val_loss):
                        raise FileNotFoundError(
                            f"Resuming with best_val_loss={best_val_loss:.6f} but {best_ckpt_path} does "
                            "not exist. Copy that run's best.pt into this folder (or set "
                            "training.run_name to the run's folder) before resuming.")
                elif best_sha is not None and self._file_fingerprint(best_ckpt_path) != best_sha:
                    raise ValueError(
                        f"{best_ckpt_path} is not the best checkpoint recorded in {resume_from} "
                        "(fingerprint mismatch: it was replaced or comes from another run). Restore "
                        "the run's own best.pt before resuming.")
                elif best_sha is None:
                    logger.warning("Checkpoint does not record which weights best.pt holds (older "
                                   "format); %s is trusted as this run's best.", best_ckpt_path)
            else:
                # The resumed weights are, by the caller's --best-val, the best so
                # far, so best.pt must hold exactly them.
                best_sha = state_dict_fingerprint(self.recon_net.state_dict())
                if os.path.exists(best_ckpt_path) and self._file_fingerprint(best_ckpt_path) != best_sha:
                    backup = os.path.join(self.ckpt_dir,
                                          f"best.superseded_{time.strftime('%Y%m%d_%H%M%S')}.pt")
                    os.replace(best_ckpt_path, backup)           # kept, never deleted
                    logger.warning("%s held different weights than the resumed ones; it was kept as "
                                   "%s and replaced by the resumed weights.", best_ckpt_path, backup)
                if not os.path.exists(best_ckpt_path):
                    self._atomic_save(self.recon_net.state_dict(), best_ckpt_path)
                    logger.info("Saved the resumed weights as %s (best_val_loss=%.6f).",
                                best_ckpt_path, best_val_loss)
            logger.info(
                "Resumed from %s  |  start_epoch=%d  best_val_loss=%.6f  patience=%d",
                resume_from, start_epoch, best_val_loss, patience_count,
            )
            if start_epoch > self.epochs:
                logger.info("All %d epochs already done; finalising only.", self.epochs)
            # Keep the per-epoch file-shuffle sequence identical to an
            # uninterrupted run.
            train_loader._epoch = start_epoch - 1
            # Restore global RNG (dropout masks) so a resume from last.pt
            # reproduces an uninterrupted run exactly.
            if rng_state is not None:
                torch.set_rng_state(rng_state["torch"].cpu())
                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["cuda"]])

        for epoch in range(start_epoch, self.epochs + 1):
            if skip_training:
                break
            train_loss = self._train_epoch(train_loader, optimiser)
            if self._last_epoch_steps != steps_per_epoch:
                logger.warning("Epoch %d ran %d optimiser steps but the lr schedule assumes %d.",
                               epoch, self._last_epoch_steps, steps_per_epoch)
            val_loss   = self._val_epoch(val_loader)
            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch} (train={train_loss}, val={val_loss}). "
                    f"Stopping; last.pt still holds epoch {epoch - 1}.")
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            ref_txt = ""
            if fixed_val:
                ref = val_loss if val_ref_loader is None else self._val_epoch(val_ref_loader)
                self.val_ref_loss_history.append(ref)
                ref_txt = f" | val_ref_loss={ref:.6f}"
            logger.info(
                "Epoch %02d/%02d | train_loss=%.6f | val_loss=%.6f | lr=%.3e%s",
                epoch, self.epochs, train_loss, val_loss, optimiser.param_groups[0]["lr"], ref_txt,
            )

            # Same rule as the source: improvement must exceed min_delta.
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_count = 0
                state = self.recon_net.state_dict()
                self._atomic_save(state, best_ckpt_path)
                best_sha = state_dict_fingerprint(state)
                logger.info("  → New best val_loss=%.6f  (checkpoint saved)", best_val_loss)
            else:
                patience_count += 1

            self._atomic_save(
                {
                    "model": self.recon_net.state_dict(),
                    "optimizer": optimiser.state_dict(),
                    "scheduler": None if self._scheduler is None else self._scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "patience_count": patience_count,
                    "history": {"train": list(self.train_loss_history),
                                "val": list(self.val_loss_history),
                                "val_ref": list(self.val_ref_loss_history)},
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                    "run_info": run_info,
                    "train_stats": self._train_stats_dict(n_train, files_fp),
                    "schedule": dict(self._schedule),       # lr curve, incl. any rebase (R1)
                    "best_sha256": best_sha,                # which weights best.pt must hold (R3)
                },
                last_ckpt_path,
            )

            # Stop after `patience` consecutive epochs without improvement
            # (patience_count > 0 keeps patience=0 meaning "stop at the first
            # non-improving epoch", as in the original trainer).
            if patience_count > 0 and patience_count >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Load best weights
        if not os.path.exists(best_ckpt_path):
            raise RuntimeError(f"No best checkpoint at {best_ckpt_path}: no epoch improved on "
                               f"best_val_loss={best_val_loss}. last.pt is kept.")
        best_state = torch.load(best_ckpt_path, map_location=self.device, weights_only=True)
        if best_sha is not None and state_dict_fingerprint(best_state) != best_sha:
            raise RuntimeError(f"{best_ckpt_path} changed during the run (fingerprint mismatch); "
                               "refusing to fit norm_stats.pt on unknown weights. last.pt is kept.")
        self.recon_net.load_state_dict(best_state)
        self.recon_net.eval()

        # 4. Compute p5/p95 of train loss  ----------------------------------
        logger.info("Computing p5/p95 of per-patch train loss…")
        L_all = self._compute_train_losses(train_scan)
        norm_stats = self._compute_norm_stats(L_all)

        # 5. Save NormStats  ------------------------------------------------
        norm_stats_path = os.path.join(self.ckpt_dir, "norm_stats.pt")
        norm_stats.save(norm_stats_path)

        return self.recon_net, norm_stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _streaming_mean_std(self, loader) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        One pass over ``loader`` accumulating float64 sums / sums of squares,
        so memory stays O(dim) instead of O(N * dim).

        Returns (E_mean, E_std, T_mean, T_std, n) with tensors on CPU (float32).
        Uses the unbiased (N-1) variance, matching Tensor.std().
        """
        sE = sE2 = sT = sT2 = None
        n = 0
        for e, t in loader:
            e = e.double()
            t = t.double()
            if sE is None:
                sE  = torch.zeros(e.shape[1], dtype=torch.float64, device=e.device)
                sE2 = torch.zeros_like(sE)
                sT  = torch.zeros(t.shape[1], dtype=torch.float64, device=t.device)
                sT2 = torch.zeros_like(sT)
            sE  += e.sum(0)
            sE2 += (e * e).sum(0)
            sT  += t.sum(0)
            sT2 += (t * t).sum(0)
            n   += e.shape[0]

        def finish(s, s2):
            mean = s / n
            var = (s2 - n * mean * mean) / max(n - 1, 1)
            std = var.clamp(min=0).sqrt().clamp(min=1e-8)
            return mean.float().cpu(), std.float().cpu()

        E_mean, E_std = finish(sE, sE2)
        T_mean, T_std = finish(sT, sT2)
        return E_mean, E_std, T_mean, T_std, n

    # ------------------------------------------------------------------
    def _standardise(
        self, E: Tensor, T: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply fitted standardisation to a batch. Returns tensors on self.device."""
        eps = 1e-8
        dev = self.device
        E_std = (E.to(dev) - self._E_mean.to(dev)) / (self._E_std_v.to(dev) + eps)
        T_std = (T.to(dev) - self._T_mean.to(dev)) / (self._T_std_v.to(dev) + eps)
        return E_std, T_std

    # ------------------------------------------------------------------
    def _train_epoch(
        self, loader: DataLoader, optimiser: torch.optim.Optimizer
    ) -> float:
        """
        Run one training epoch.

        Returns
        -------
        mean_loss : float  average MSE loss over all patches in this epoch
        """
        self.recon_net.train()
        total_loss = 0.0
        total_n    = 0
        steps      = 0

        for e_batch, t_batch in loader:
            e_batch = e_batch.float()
            t_batch = t_batch.float()
            E_std, T_std = self._standardise(e_batch, t_batch)

            E_hat = self.recon_net(T_std)                     # (B, dv)
            # "sum" = original LADP objective; "mean" = nn.MSELoss() as in the
            # source (matters for grad clipping, whose threshold is absolute).
            loss  = F.mse_loss(E_hat, E_std, reduction=self.loss_reduction)

            optimiser.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.recon_net.parameters(), self.grad_clip)
            optimiser.step()
            if self._scheduler is not None:
                self._scheduler.step()

            n = E_std.numel()             # B * dv
            total_loss += loss.item() * (n if self.loss_reduction == "mean" else 1)
            total_n    += n
            steps      += 1

        self._last_epoch_steps = steps
        return total_loss / max(total_n, 1)

    # ------------------------------------------------------------------
    def _val_epoch(self, loader: DataLoader) -> float:
        """
        Run one validation epoch.

        Returns
        -------
        mean_loss : float  average MSE loss over all patches in this epoch
        """
        self.recon_net.eval()
        total_loss = 0.0
        total_n    = 0

        with torch.no_grad():
            for e_batch, t_batch in loader:
                e_batch = e_batch.float()
                t_batch = t_batch.float()
                E_std, T_std = self._standardise(e_batch, t_batch)

                E_hat = self.recon_net(T_std)
                loss  = F.mse_loss(E_hat, E_std, reduction="sum")

                total_loss += loss.item()
                total_n    += E_std.numel()

        return total_loss / max(total_n, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_train_losses(self, loader: DataLoader) -> Tensor:
        """
        Compute per-patch MSE loss on the train split.

        Returns
        -------
        L_all : Tensor of shape (N_train,)
        """
        self.recon_net.eval()
        L_parts: List[Tensor] = []

        for e_batch, t_batch in loader:
            e_batch = e_batch.float()
            t_batch = t_batch.float()
            E_std, T_std = self._standardise(e_batch, t_batch)

            E_hat = self.recon_net(T_std)                    # (B, dv)
            L_i   = (E_std - E_hat).pow(2).mean(dim=-1)     # (B,) per-patch MSE
            L_parts.append(L_i.cpu())

        return torch.cat(L_parts, dim=0)   # (N_train,)

    # ------------------------------------------------------------------
    def _compute_norm_stats(self, L_all: Tensor) -> NormStats:
        """
        Build a NormStats from the fitted mean/std and per-patch train losses.

        Parameters
        ----------
        L_all : Tensor of shape (N,)      per-patch MSE on train split

        Returns
        -------
        NormStats
        """
        L_all = L_all.float()
        # torch.quantile caps input size (2**24); kthvalue has no such limit.
        n = L_all.numel()
        k5  = max(1, min(n, int(round(0.05 * (n - 1))) + 1))
        k95 = max(1, min(n, int(round(0.95 * (n - 1))) + 1))
        p5  = float(L_all.kthvalue(k5).values.item())
        p95 = float(L_all.kthvalue(k95).values.item())

        return NormStats(
            E_mean=self._E_mean.cpu(),
            E_std=self._E_std_v.cpu(),
            T_mean=self._T_mean.cpu(),
            T_std=self._T_std_v.cpu(),
            L_p5=p5,
            L_p95=p95,
            # fitted with the weights now in recon_net (= best.pt): record them (M4)
            meta=norm_stats_meta(self.recon_net, os.path.basename(self.ckpt_dir)),
        )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("LADP loss_estimator.py — shape sanity checks")
    print("=" * 60)

    # Dimensions matching the LADP config
    DV    = 1024   # pre-connector CLIP features (ViT-L/14 patch dim)
    DT    = 4096   # post-connector LLM tokens (LLaVA-1.5-7B hidden dim)
    HIDDEN = 2048
    B     = 16     # batch size

    # ------------------------------------------------------------------
    print("\n[1] ReconstructionNet variants")
    for depth in ("mlp1", "mlp2", "mlp3"):
        net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=HIDDEN, depth=depth)
        x   = torch.randn(B, DT)
        out = net(x)
        assert out.shape == (B, DV), f"Expected ({B}, {DV}), got {out.shape}"
        print(f"  {depth:5s} | params={net.param_count():>12,d} | output={tuple(out.shape)}")

    # ------------------------------------------------------------------
    print("\n[2] NormStats  save / load")
    import tempfile, pathlib
    ns = NormStats(
        E_mean=torch.zeros(DV),
        E_std=torch.ones(DV),
        T_mean=torch.zeros(DT),
        T_std=torch.ones(DT),
        L_p5=0.01,
        L_p95=0.99,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "norm_stats.pt")
        ns.save(path)
        ns2 = NormStats.load(path)
    assert ns2.L_p5  == ns.L_p5
    assert ns2.L_p95 == ns.L_p95
    assert ns2.E_mean.shape == (DV,)
    assert ns2.T_mean.shape == (DT,)
    print("  NormStats  save/load ✓")

    # ------------------------------------------------------------------
    print("\n[3] LossEstimator  (all distance modes)")
    net = ReconstructionNet(d_in=DT, d_out=DV, depth="mlp3")
    ns_dummy = NormStats(
        E_mean=torch.zeros(DV),
        E_std=torch.ones(DV),
        T_mean=torch.zeros(DT),
        T_std=torch.ones(DT),
        L_p5=0.0,
        L_p95=1.0,
    )

    E_raw = torch.randn(B, DV)
    T_raw = torch.randn(B, DT)

    for dist in ("mse", "norm_relative", "cosine"):
        est = LossEstimator(net, ns_dummy, distance=dist)
        L   = est.compute(E_raw, T_raw)
        assert L.shape == (B,), f"Expected ({B},), got {L.shape}"
        L_norm = est.compute_normalised(E_raw, T_raw, norm_mode="robust")
        assert L_norm.shape == (B,)
        assert L_norm.min() >= 0.0 and L_norm.max() <= 1.0, \
            f"L_norm out of [0,1]: min={L_norm.min():.4f}, max={L_norm.max():.4f}"
        L_pi = est.compute_normalised(E_raw, T_raw, norm_mode="per_image")
        assert L_pi.shape == (B,)
        assert L_pi.min() >= 0.0 and L_pi.max() <= 1.0 + 1e-6
        print(f"  dist={dist:14s} | L.shape={tuple(L.shape)} | L∈[{L.min():.3f},{L.max():.3f}]")

    # ------------------------------------------------------------------
    print("\n[4] PatchDataset + ReconstructionTrainer (tiny synthetic run)")
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmp:
        # Write 3 synthetic feature files
        N_per_file = 20
        for i in range(3):
            torch.save(
                {
                    "E": torch.randn(N_per_file, DV),
                    "T": torch.randn(N_per_file, DT),
                },
                os.path.join(tmp, f"features_{i}.pt"),
            )

        ds = PatchDataset(tmp)
        assert len(ds) == 3 * N_per_file, f"Expected {3*N_per_file}, got {len(ds)}"
        e0, t0 = ds[0]
        assert e0.shape == (DV,) and t0.shape == (DT,)
        e_last, t_last = ds[len(ds) - 1]
        assert e_last.shape == (DV,) and t_last.shape == (DT,)
        print(f"  PatchDataset len={len(ds)} | e[0]={tuple(e0.shape)} | t[0]={tuple(t0.shape)}")

        # Tiny trainer run (1 epoch, small net)
        tiny_net = ReconstructionNet(d_in=DT, d_out=DV, hidden_dim=64, depth="mlp2")
        cfg = {
            "checkpoints_dir": tmp,
            "training": {
                "epochs": 2,
                "lr": 1e-3,
                "batch_size": 10,
                "patience": 2,
                "val_fraction": 0.3,
                "num_workers": 0,
                "seed": 0,
            },
        }
        trainer = ReconstructionTrainer(tiny_net, cfg, device="cpu")
        best_net, stats = trainer.fit(tmp)
        assert isinstance(best_net, ReconstructionNet)
        assert isinstance(stats, NormStats)
        assert stats.E_mean.shape == (DV,)
        assert stats.T_mean.shape == (DT,)
        assert stats.L_p5 <= stats.L_p95
        print(f"  Trainer fit ✓ | p5={stats.L_p5:.6f} | p95={stats.L_p95:.6f}")

    print("\nAll sanity checks passed.")
