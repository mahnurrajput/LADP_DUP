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
import math
import logging
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
    mlp1  : linear(d_in, d_out)                     — no hidden layer
    mlp2  : linear(d_in, hidden) → GELU → linear(hidden, d_out)
    mlp3  : linear(d_in, hidden) → GELU → Dropout →
             linear(hidden, hidden) → GELU → Dropout →
             linear(hidden, d_out)            [DEFAULT]

    Parameters
    ----------
    d_in      : int   input dimension  dt  (post-connector tokens)   default 4096
    d_out     : int   output dimension dv  (pre-connector features)  default 1024
    hidden_dim: int   width of hidden layers                         default 2048
    depth     : str   one of {"mlp1", "mlp2", "mlp3"}               default "mlp3"
    dropout   : float dropout probability between hidden layers      default 0.1

    Forward
    -------
    T : (B, d_in)  standardised post-connector tokens
    → Ê : (B, d_out)  predicted pre-connector features (unconstrained)
    """

    def __init__(
        self,
        d_in: int = 4096,
        d_out: int = 1024,
        hidden_dim: int = 2048,
        depth: str = "mlp3",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth not in {"mlp1", "mlp2", "mlp3"}:
            raise ValueError(f"depth must be 'mlp1', 'mlp2', or 'mlp3'; got {depth!r}")

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

        else:  # mlp3 (default)
            # two hidden layers
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

        self.net = nn.Sequential(*layers)
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

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise to disk via torch.save."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "E_mean": self.E_mean,
                "E_std":  self.E_std,
                "T_mean": self.T_mean,
                "T_std":  self.T_std,
                "L_p5":   self.L_p5,
                "L_p95":  self.L_p95,
            },
            path,
        )
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
        )


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
        self.lr          = float(training_cfg.get("lr", 1e-4))
        self.batch_size  = int(training_cfg.get("batch_size", 128))
        self.epochs      = int(training_cfg.get("epochs", 30))
        self.patience    = int(training_cfg.get("patience", 5))
        self.val_frac    = float(training_cfg.get("val_fraction", 0.05))
        self.num_workers = int(training_cfg.get("num_workers", 4))
        self.seed        = int(training_cfg.get("seed", 42))

        self.ckpt_dir = os.path.join(
            config.get("checkpoints_dir", "checkpoints"), "recon"
        )

    # ------------------------------------------------------------------
    def fit(self, cache_dir: str) -> Tuple[ReconstructionNet, NormStats]:
        """
        Train gφ and return the best model + fitted NormStats.

        Parameters
        ----------
        cache_dir : str  directory with features_{i}.pt files

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
        if len(file_paths) < 2:
            raise ValueError("Need at least 2 cached images for a train/val split")
        perm = torch.randperm(
            len(file_paths), generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        n_val = min(len(file_paths) - 1, max(1, int(len(file_paths) * self.val_frac)))
        val_files   = sorted(file_paths[i] for i in perm[:n_val])
        train_files = sorted(file_paths[i] for i in perm[n_val:])
        logger.info("Train images: %d  |  Val images: %d", len(train_files), len(val_files))

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
        # Unshuffled pass over train files (stats / p5-p95 do not need shuffling)
        train_scan = ShardedPatchLoader(
            train_files, 8192, files_per_shard,
            shuffle=False, seed=self.seed, device=self.device,
        )

        # 2. Streaming standardisation stats from train split  -------------
        logger.info("Computing normalisation statistics on train split (streaming)…")
        (self._E_mean, self._E_std_v,
         self._T_mean, self._T_std_v, n_train) = self._streaming_mean_std(train_scan)
        logger.info("Train patches: %d", n_train)

        # 3. Train  ---------------------------------------------------------
        optimiser = torch.optim.Adam(self.recon_net.parameters(), lr=self.lr)

        best_val_loss  = float("inf")
        patience_count = 0
        os.makedirs(self.ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(self.ckpt_dir, "best.pt")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimiser)
            val_loss   = self._val_epoch(val_loader)
            logger.info(
                "Epoch %02d/%02d | train_loss=%.6f | val_loss=%.6f",
                epoch, self.epochs, train_loss, val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save(self.recon_net.state_dict(), best_ckpt_path)
                logger.info("  → New best val_loss=%.6f  (checkpoint saved)", best_val_loss)
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Load best weights
        self.recon_net.load_state_dict(
            torch.load(best_ckpt_path, map_location=self.device, weights_only=True)
        )
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

        for e_batch, t_batch in loader:
            e_batch = e_batch.float()
            t_batch = t_batch.float()
            E_std, T_std = self._standardise(e_batch, t_batch)

            E_hat = self.recon_net(T_std)                     # (B, dv)
            loss  = F.mse_loss(E_hat, E_std, reduction="sum") # scalar (sum over B*dv)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            total_n    += E_std.numel()   # B * dv

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
