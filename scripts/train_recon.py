"""
train_recon.py — LADP project
================================
Trains the reconstruction network gφ (ReconstructionNet) on cached COCO
features.  This is the ONLY learnable training step in the entire LADP
pipeline — the LLaVA backbone and CLIP scorer are both frozen.

Usage
-----
  python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml

Prerequisites
-------------
  - cache_features.py must have been run for the coco_train split first:
      python scripts/cache_features.py --config configs/base_llava_vizwiz.yaml \\
                                        --split coco_train

Resuming
--------
  --resume <ckpt_dir>/last.pt                      full state (recommended)
  --resume <ckpt_dir>/best.pt --start-epoch N --best-val X   weights only

Outputs
-------
  <checkpoints_dir>/<training.run_name>/best.pt        — best weights (state_dict)
  <checkpoints_dir>/<training.run_name>/last.pt        — full resume state, every epoch
  <checkpoints_dir>/<training.run_name>/norm_stats.pt  — NormStats (mean/std/p5/p95)
  (defaults: checkpoints_dir = checkpoints, run_name = recon)
  <output_dir>/config_snapshot.yaml  — copy of the config used for this run
  <output_dir>/train_recon.log       — full log
  <output_dir>/recon_loss_curve.png  — training/val loss plot
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless GPU servers
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Allow running from project root or from the scripts/ subdirectory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from loss_estimator import (  # noqa: E402
    ReconstructionTrainer,
    NormStats,
    build_recon_net_from_config,
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    """Write logs to both the console and <output_dir>/train_recon.log.

    Call this before any logging statements.  The file is opened in append
    mode so re-runs accumulate rather than clobber history.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_recon.log"

    fmt     = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy third-party loggers.
    for noisy in ("transformers", "PIL", "urllib3", "filelock", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Log file: %s", log_path)


# ---------------------------------------------------------------------------
# Loss curve plot
# ---------------------------------------------------------------------------

def plot_loss_curve(
    train_losses: list[float],
    val_losses:   list[float],
    output_dir:   Path,
) -> Path:
    """Save a training/validation loss curve PNG to output_dir.

    Parameters
    ----------
    train_losses : list[float]
        Mean train MSE loss per epoch (length = number of epochs run).
    val_losses : list[float]
        Mean val MSE loss per epoch (same length).
    output_dir : Path
        Directory in which to save ``recon_loss_curve.png``.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train MSE", linewidth=1.8, color="#2196F3")
    ax.plot(epochs, val_losses,   label="Val MSE",   linewidth=1.8, color="#F44336", linestyle="--")

    # Mark the epoch with the best (minimum) val loss.
    if val_losses:
        best_epoch = int(min(range(len(val_losses)), key=lambda i: val_losses[i])) + 1
        best_val   = min(val_losses)
        ax.axvline(best_epoch, color="#4CAF50", linestyle=":", linewidth=1.4,
                   label=f"Best val epoch={best_epoch}  loss={best_val:.5f}")
        ax.scatter([best_epoch], [best_val], color="#4CAF50", zorder=5, s=60)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss (standardised space)", fontsize=12)
    ax.set_title("LADP — Reconstruction Network gφ Training", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    out_path = output_dir / "recon_loss_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("Loss curve saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def save_config_snapshot(cfg: dict, output_dir: Path) -> None:
    """Write a YAML copy of the config used for this training run."""
    import yaml
    out_path = output_dir / "config_snapshot.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logging.info("Config snapshot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Sanity check after training
# ---------------------------------------------------------------------------

def sanity_check(
    ckpt_path:       Path,
    norm_stats_path: Path,
    cfg:             dict,
    device:          torch.device,
) -> bool:
    """Load the saved checkpoint and verify tensor shapes.

    Returns True if everything is as expected.  On failure it logs the error
    and returns False; main() then exits with a non-zero code.
    """
    logging.info("Running post-training sanity check…")

    try:
        net = build_recon_net_from_config(cfg)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        net.load_state_dict(state)
        net.to(device).eval()

        # Forward pass on a random batch.
        dummy_T = torch.randn(8, 4096, device=device)
        with torch.no_grad():
            dummy_E_hat = net(dummy_T)
        assert dummy_E_hat.shape == (8, 1024), (
            f"Expected output shape (8, 1024), got {dummy_E_hat.shape}"
        )
        logging.info("  Checkpoint loaded and forward pass OK.  Output shape: %s",
                     tuple(dummy_E_hat.shape))

        ns = NormStats.load(str(norm_stats_path))
        assert ns.E_mean.shape == (1024,), f"E_mean shape: {ns.E_mean.shape}"
        assert ns.T_mean.shape == (4096,), f"T_mean shape: {ns.T_mean.shape}"
        assert ns.L_p5 <= ns.L_p95, (
            f"p5={ns.L_p5:.6f} > p95={ns.L_p95:.6f} — unexpected ordering"
        )
        logging.info("  NormStats loaded OK.  E_mean: %s  T_mean: %s",
                     tuple(ns.E_mean.shape), tuple(ns.T_mean.shape))
        logging.info("Sanity check passed.")
        return True

    except Exception as exc:
        logging.error("Sanity check FAILED: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the LADP reconstruction network gφ on cached COCO features.  "
            "Run cache_features.py --split coco_train first."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the LADP config YAML, e.g. configs/base_llava_vizwiz.yaml",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume from: checkpoints/recon/last.pt (full state) "
             "or checkpoints/recon/best.pt (weights only; needs --start-epoch "
             "and --best-val).",
    )
    parser.add_argument("--start-epoch", type=int, default=None,
                        help="First epoch to run when resuming from weights only.")
    parser.add_argument("--best-val", type=float, default=None,
                        help="Best val loss so far when resuming from weights only.")
    args = parser.parse_args()
    if not args.resume and (args.start_epoch is not None or args.best_val is not None):
        parser.error("--start-epoch / --best-val are only valid together with --resume")

    # ------------------------------------------------------------------
    # Load config.
    # ------------------------------------------------------------------
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"ERROR: Config file not found: {config_path}")

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Resolve key directories.
    # ------------------------------------------------------------------
    project_root = config_path.parent.parent  # ladp/

    output_dir_raw = cfg.get("output_dir", "results/stage1_diagnostic")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    cache_dir_raw = cfg.get("data", {}).get("cache_dir", "data/cache")
    cache_dir = Path(cache_dir_raw)
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    coco_cache_dir = cache_dir / "coco_train"

    # Checkpoints go to <checkpoints_dir>/<training.run_name>/ (default
    # checkpoints/recon/) so pilot and new runs never overwrite each other.
    checkpoints_dir = Path(cfg.get("checkpoints_dir", "checkpoints"))
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = project_root / checkpoints_dir

    # ------------------------------------------------------------------
    # Setup logging early so every subsequent step is captured.
    # ------------------------------------------------------------------
    setup_logging(output_dir)

    logging.info("=" * 65)
    logging.info("LADP train_recon.py")
    logging.info("=" * 65)
    logging.info("Config          : %s", config_path)
    logging.info("COCO cache dir  : %s", coco_cache_dir)
    logging.info("Output dir      : %s", output_dir)
    logging.info("Checkpoints dir : %s", checkpoints_dir)

    # ------------------------------------------------------------------
    # Validate that the COCO cache exists.
    # ------------------------------------------------------------------
    if not coco_cache_dir.exists():
        logging.error(
            "COCO cache not found — run cache_features.py first.\n"
            "  Expected: %s\n"
            "  Command : python scripts/cache_features.py "
            "--config %s --split coco_train",
            coco_cache_dir,
            args.config,
        )
        sys.exit(1)

    # Look for at least one .pt feature file.
    feature_files = list(coco_cache_dir.glob("features_*.pt"))
    if not feature_files:
        logging.error(
            "No features_*.pt files found in %s.\n"
            "The cache directory exists but appears empty.  "
            "Re-run cache_features.py for the coco_train split.",
            coco_cache_dir,
        )
        sys.exit(1)

    logging.info("Found %d cached feature file(s) in %s",
                 len(feature_files), coco_cache_dir)

    # ------------------------------------------------------------------
    # Random seed.
    # ------------------------------------------------------------------
    seed = int(cfg.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Random seed: %d", seed)

    # ------------------------------------------------------------------
    # Device.
    # ------------------------------------------------------------------
    device_str = cfg.get("model", {}).get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "CUDA requested in config but not available.  Falling back to CPU."
        )
        device_str = "cpu"
    device = torch.device(device_str)
    logging.info("Training device: %s", device)

    # ------------------------------------------------------------------
    # Build ReconstructionNet from config.
    # ------------------------------------------------------------------
    # loss.reconstructor is required (no silent fallback to the pilot model)
    # and loss.expected_params, if set, must match the built model.
    try:
        recon_net = build_recon_net_from_config(cfg)
    except ValueError as exc:
        logging.error("Invalid model config: %s", exc)
        sys.exit(2)
    n_params = recon_net.param_count()
    logging.info(
        "ReconstructionNet: depth=%s  hidden_dim=%d  dropout=%.2f  params=%s",
        recon_net.depth, recon_net.hidden_dim, recon_net.dropout_p, f"{n_params:,}",
    )

    # ------------------------------------------------------------------
    # Save config snapshot before anything can go wrong.
    # ------------------------------------------------------------------
    save_config_snapshot(cfg, output_dir)

    # ------------------------------------------------------------------
    # Build trainer — inject checkpoints_dir into config copy.
    # ------------------------------------------------------------------
    cfg_for_trainer = dict(cfg)
    cfg_for_trainer["checkpoints_dir"] = str(checkpoints_dir)
    fv = cfg.get("training", {}).get("fixed_val_images")
    if fv and not Path(fv).is_absolute():
        cfg_for_trainer["training"] = dict(cfg["training"], fixed_val_images=str(project_root / fv))

    trainer = ReconstructionTrainer(recon_net, cfg_for_trainer, device=device_str)

    # ------------------------------------------------------------------
    # Train.
    # ------------------------------------------------------------------
    fit_kwargs: dict = {}
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            logging.error("Resume checkpoint not found: %s", resume_path)
            sys.exit(1)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        full = isinstance(ckpt, dict) and "model" in ckpt and "optimizer" in ckpt
        if full:
            if args.start_epoch is not None or args.best_val is not None:
                logging.warning("--start-epoch / --best-val are ignored: %s is a full "
                                "checkpoint and carries its own epoch and best loss.",
                                resume_path.name)
            start_epoch = int(ckpt["epoch"]) + 1
            fit_kwargs = {"resume_from": str(resume_path)}
        else:
            if args.start_epoch is None or args.best_val is None:
                logging.error("%s holds weights only: pass both --start-epoch and "
                              "--best-val (or resume from last.pt).", resume_path.name)
                sys.exit(1)
            start_epoch = args.start_epoch
            fit_kwargs = {"resume_from": str(resume_path),
                          "start_epoch": args.start_epoch,
                          "best_val_loss": args.best_val}
        has_history = full and "history" in ckpt
        del ckpt
        if not has_history and start_epoch > 1:
            # Older checkpoints carry no loss history: recover epochs
            # 1..start_epoch-1 from the log, using only the most recent run
            # that logged them (later lines win) and refusing gaps.
            import re
            prev: dict[int, tuple[float, float]] = {}
            log_path = output_dir / "train_recon.log"
            pat = re.compile(r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| val_loss=([\d.]+)")
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    m = pat.search(line)
                    if m:
                        ep = int(m.group(1))
                        if ep == 1:
                            prev = {}       # a new run started: forget older runs
                        prev[ep] = (float(m.group(2)), float(m.group(3)))
            wanted = list(range(1, start_epoch))
            if all(ep in prev for ep in wanted):
                for ep in wanted:
                    trainer.train_loss_history.append(prev[ep][0])
                    trainer.val_loss_history.append(prev[ep][1])
            else:
                missing = [ep for ep in wanted if ep not in prev]
                logging.warning("Could not recover epochs %s from %s; the loss curve "
                                "will cover only the resumed epochs.", missing, log_path)
        logging.info("Resuming from %s at epoch %d (%d earlier epoch(s) in history)",
                     resume_path, start_epoch, len(trainer.train_loss_history))

    logging.info("Starting training…")
    t0 = time.time()

    try:
        best_net, norm_stats = trainer.fit(str(coco_cache_dir), **fit_kwargs)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # Includes the refusals added for safety: incomplete / legacy cache,
        # wrong image count, resume mismatch, missing best.pt, non-finite loss.
        logging.error("Training stopped: %s", exc, exc_info=True)
        sys.exit(2)

    elapsed = time.time() - t0
    logging.info("Training complete in %.1f s  (%.1f min)", elapsed, elapsed / 60)

    # ------------------------------------------------------------------
    # Identify saved artefact paths (ReconstructionTrainer writes these).
    # ------------------------------------------------------------------
    ckpt_dir        = Path(trainer.ckpt_dir)
    ckpt_path       = ckpt_dir / "best.pt"
    norm_stats_path = ckpt_dir / "norm_stats.pt"

    # ------------------------------------------------------------------
    # Plot loss curve.
    # ------------------------------------------------------------------
    if trainer.train_loss_history:
        plot_loss_curve(trainer.train_loss_history, trainer.val_loss_history, output_dir)
    else:
        logging.warning(
            "No per-epoch loss history was recorded — loss curve will not be generated.  "
            "This is unexpected; check ReconstructionTrainer.fit()."
        )

    # ------------------------------------------------------------------
    # Post-training sanity check.
    # ------------------------------------------------------------------
    if ckpt_path.exists() and norm_stats_path.exists():
        sanity_ok = sanity_check(ckpt_path, norm_stats_path, cfg, device)
    else:
        sanity_ok = False
        logging.error(
            "Expected checkpoint or norm_stats not found after training.  "
            "ckpt=%s  norm_stats=%s",
            ckpt_path, norm_stats_path,
        )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    best_val = min(trainer.val_loss_history) if trainer.val_loss_history else float("nan")
    n_epochs_run = len(trainer.train_loss_history)

    logging.info("=" * 65)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 65)
    if trainer.val_ref_loss_history:
        logging.info("  Best val_ref MSE    : %.6f  (fixed reference images; comparable with the pilot)",
                     min(trainer.val_ref_loss_history))
    logging.info("  Epochs run          : %d / %d  (early stopping may apply)",
                 n_epochs_run, trainer.epochs)
    logging.info("  Best val MSE loss   : %.6f", best_val)
    logging.info("  NormStats p5 / p95  : %.6f / %.6f",
                 norm_stats.L_p5, norm_stats.L_p95)
    logging.info("  Trainable params    : %s", f"{n_params:,}")
    logging.info("  Training time       : %.1f s  (%.1f min)", elapsed, elapsed / 60)
    logging.info("  Checkpoint          : %s", ckpt_path)
    logging.info("  NormStats           : %s", norm_stats_path)
    logging.info("  Config snapshot     : %s", output_dir / "config_snapshot.yaml")
    logging.info("  Loss curve          : %s", output_dir / "recon_loss_curve.png")

    # Interpretation hint for p5/p95 values.
    logging.info(
        "\n  Loss distribution note:\n"
        "    p5  = %.6f  (5th  percentile per-patch MSE on train set)\n"
        "    p95 = %.6f  (95th percentile per-patch MSE on train set)\n"
        "    At inference, per-patch losses outside this range are clipped\n"
        "    to [0, 1] by the robust normalisation mode.",
        norm_stats.L_p5, norm_stats.L_p95,
    )

    if not sanity_ok:
        logging.error("Post-training sanity check failed; exiting with code 3.")
        sys.exit(3)
    logging.info("Done.  Next step: run the VizWiz evaluation pipeline.")


if __name__ == "__main__":
    main()
