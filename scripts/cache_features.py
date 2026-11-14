"""
cache_features.py — LADP project
==================================
Pre-computes and saves E (pre-connector CLIP features) and T (post-connector
projected tokens) from LLaVA-1.5-7B for every image in a configured directory.

Run ONCE before training.  The heavy backbone forward pass is done here so
that train_recon.py never has to touch the 7B model — it only reads .pt files.

Usage
-----
  python scripts/cache_features.py --config configs/base_llava_vizwiz.yaml \\
                                    --split coco_train

  python scripts/cache_features.py --config configs/base_llava_vizwiz.yaml \\
                                    --split vizwiz_val

  # dry-run / smoke-test on a handful of images:
  python scripts/cache_features.py --config configs/base_llava_vizwiz.yaml \\
                                    --split coco_train --max_images 10

Output layout
-------------
  <cache_dir>/<split>/features_000000.pt
  <cache_dir>/<split>/features_000001.pt
  ...
  <cache_dir>/<split>/manifest.json

Each .pt file contains:
  {
      "E":          Tensor[N, 1024],   # pre-connector CLIP patch features (CPU)
      "T":          Tensor[N, 4096],   # post-connector projected tokens   (CPU)
      "image_path": str,               # absolute path of the source image
      "image_idx":  int,               # index i (zero-based) in this split
  }
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running from project root or from the scripts/ subdirectory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from connector_adapter import LlavaAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    """Write to both console (INFO) and a file in output_dir (DEBUG)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "cache_features.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy third-party loggers at DEBUG level.
    for noisy in ("transformers", "PIL", "urllib3", "filelock", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Logging to %s", log_path)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def discover_images(image_dir: Path, max_images: int | None = None) -> list[Path]:
    """Return a sorted list of image file paths under image_dir.

    Parameters
    ----------
    image_dir : Path
        Directory to search (non-recursive — COCO / VizWiz are flat).
    max_images : int | None
        If set, truncate the returned list to this many images.

    Raises
    ------
    FileNotFoundError
        If image_dir does not exist.
    RuntimeError
        If no matching image files are found.
    """
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            "Check that the path in your config YAML is correct and that the "
            "dataset has been downloaded."
        )

    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        raise RuntimeError(
            f"No image files found in {image_dir}.\n"
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    logging.info("Found %d images in %s", len(paths), image_dir)

    if max_images is not None:
        paths = paths[:max_images]
        logging.info("Limiting to first %d images (--max_images flag).", max_images)

    return paths


# ---------------------------------------------------------------------------
# GPU memory logging helper
# ---------------------------------------------------------------------------

def log_gpu_memory(tag: str = "") -> None:
    """Log current GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    logging.debug(
        "GPU memory %s — allocated: %.2f GiB | reserved: %.2f GiB",
        tag, allocated, reserved,
    )


# ---------------------------------------------------------------------------
# Core caching logic
# ---------------------------------------------------------------------------

def cache_split(
    adapter: LlavaAdapter,
    image_paths: list[Path],
    out_dir: Path,
) -> list[dict]:
    """Extract and cache features for a list of images.

    Parameters
    ----------
    adapter : LlavaAdapter
        Loaded model wrapper.
    image_paths : list[Path]
        Ordered list of images to process.
    out_dir : Path
        Directory where .pt files are written.

    Returns
    -------
    list[dict]
        Manifest entries for every successfully cached image.
        Each entry has keys: "index", "filename", "image_path", "E_shape", "T_shape".
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    errors: list[dict] = []

    log_gpu_memory("before extraction loop")

    pbar = tqdm(image_paths, desc="Caching features", unit="img", dynamic_ncols=True)

    for i, img_path in enumerate(pbar):
        out_file = out_dir / f"features_{i:06d}.pt"

        # ------------------------------------------------------------------
        # Resume support: skip already-cached files.
        # ------------------------------------------------------------------
        if out_file.exists():
            # Validate the existing file is complete (not a partial write).
            try:
                existing = torch.load(out_file, map_location="cpu", weights_only=True)
                if "E" in existing and "T" in existing:
                    manifest.append({
                        "index":      i,
                        "filename":   out_file.name,
                        "image_path": str(img_path),
                        "E_shape":    list(existing["E"].shape),
                        "T_shape":    list(existing["T"].shape),
                        "cached":     True,
                    })
                    pbar.set_postfix(status="skipped (cached)")
                    continue
            except Exception:
                # Corrupt file — overwrite it.
                logging.warning("Corrupt cache file detected, re-extracting: %s", out_file)

        # ------------------------------------------------------------------
        # Extract features.
        # ------------------------------------------------------------------
        try:
            image = Image.open(img_path)
            features = adapter.extract_features(image)

            E: torch.Tensor = features["E"].cpu()   # (N, 1024)  move to CPU before saving
            T: torch.Tensor = features["T"].cpu()   # (N, 4096)

            payload = {
                "E":          E,
                "T":          T,
                "image_path": str(img_path),
                "image_idx":  i,
            }
            torch.save(payload, out_file)

            manifest.append({
                "index":      i,
                "filename":   out_file.name,
                "image_path": str(img_path),
                "E_shape":    list(E.shape),
                "T_shape":    list(T.shape),
                "cached":     False,
            })
            pbar.set_postfix(status="ok", E=str(tuple(E.shape)), T=str(tuple(T.shape)))

        except Exception as exc:  # noqa: BLE001
            logging.error(
                "Failed to process image %d (%s): %s — skipping.",
                i, img_path, exc,
                exc_info=True,
            )
            errors.append({"index": i, "image_path": str(img_path), "error": str(exc)})
            pbar.set_postfix(status="ERROR")
            continue

        # ------------------------------------------------------------------
        # Periodic progress logging.
        # ------------------------------------------------------------------
        if (i + 1) % 100 == 0:
            logging.info(
                "Progress: %d / %d images processed  (%d errors so far)",
                i + 1, len(image_paths), len(errors),
            )
            log_gpu_memory(f"after image {i + 1}")

    # Flush tqdm so the final line doesn't get interleaved with logging.
    pbar.close()

    if errors:
        logging.warning(
            "%d image(s) failed and were skipped.  Error details above.", len(errors)
        )

    return manifest, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-cache LLaVA-1.5-7B visual features (E, T) for all images "
            "in a configured directory.  Run once before train_recon.py."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the LADP config YAML, e.g. configs/base_llava_vizwiz.yaml",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["coco_train", "vizwiz_val"],
        help=(
            "Which image directory to process.  "
            "'coco_train' → data.coco_train_images in config.  "
            "'vizwiz_val' → data.vizwiz_images in config."
        ),
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit processing to this many images.  Useful for smoke-testing.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config.
    # ------------------------------------------------------------------
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"ERROR: Config file not found: {config_path}")

    import yaml  # deferred so we give a clear error above first
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Resolve directories from config.
    # ------------------------------------------------------------------
    project_root = config_path.parent.parent  # ladp/

    data_cfg = cfg.get("data", {})
    cache_dir = Path(data_cfg.get("cache_dir", "data/cache"))
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    split_to_key = {
        "coco_train": "coco_train_images",
        "vizwiz_val": "vizwiz_images",
    }
    img_dir_key = split_to_key[args.split]
    img_dir_raw = data_cfg.get(img_dir_key)
    if not img_dir_raw:
        sys.exit(
            f"ERROR: Config key data.{img_dir_key} is missing or empty in {config_path}"
        )

    image_dir = Path(img_dir_raw)
    if not image_dir.is_absolute():
        image_dir = project_root / image_dir

    out_dir = cache_dir / args.split

    # ------------------------------------------------------------------
    # Setup logging now that we know the output directory.
    # ------------------------------------------------------------------
    setup_logging(out_dir)

    logging.info("=" * 60)
    logging.info("LADP cache_features.py")
    logging.info("=" * 60)
    logging.info("Config      : %s", config_path)
    logging.info("Split       : %s", args.split)
    logging.info("Image dir   : %s", image_dir)
    logging.info("Cache dir   : %s", out_dir)
    logging.info("Max images  : %s", args.max_images if args.max_images else "all")

    # ------------------------------------------------------------------
    # Discover images.
    # ------------------------------------------------------------------
    try:
        image_paths = discover_images(image_dir, max_images=args.max_images)
    except (FileNotFoundError, RuntimeError) as exc:
        logging.error(str(exc))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load model.
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "llava-hf/llava-1.5-7b-hf")
    precision   = model_cfg.get("precision", "fp16")
    device      = model_cfg.get("device", "cuda")

    # Gracefully fall back to CPU if CUDA is requested but unavailable.
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "CUDA requested in config but torch.cuda.is_available() == False.  "
            "Falling back to CPU (this will be very slow)."
        )
        device = "cpu"

    logging.info("Loading LlavaAdapter: model=%s  precision=%s  device=%s",
                 model_name, precision, device)
    t0_load = time.time()

    try:
        adapter = LlavaAdapter(
            model_name=model_name,
            device=device,
            precision=precision,
        )
    except Exception as exc:
        logging.error("Failed to load LlavaAdapter: %s", exc, exc_info=True)
        sys.exit(1)

    logging.info("Model loaded in %.1f s", time.time() - t0_load)
    log_gpu_memory("after model load")

    # ------------------------------------------------------------------
    # Extract and cache.
    # ------------------------------------------------------------------
    t0_extract = time.time()
    manifest, errors = cache_split(adapter, image_paths, out_dir)
    elapsed = time.time() - t0_extract

    # ------------------------------------------------------------------
    # Write manifest.json.
    # ------------------------------------------------------------------
    manifest_path = out_dir / "manifest.json"
    manifest_data = {
        "split":        args.split,
        "image_dir":    str(image_dir),
        "cache_dir":    str(out_dir),
        "total_images": len(image_paths),
        "cached_ok":    len(manifest),
        "errors":       len(errors),
        "elapsed_s":    round(elapsed, 1),
        "files":        manifest,
        "failed":       errors,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    logging.info("Manifest written to %s", manifest_path)

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("Done.")
    logging.info("  Images found   : %d", len(image_paths))
    logging.info("  Cached OK      : %d", len(manifest))
    logging.info("  Errors skipped : %d", len(errors))
    logging.info("  Elapsed        : %.1f s  (%.2f img/s)",
                 elapsed, len(manifest) / max(elapsed, 1e-6))
    logging.info("  Output dir     : %s", out_dir)
    if errors:
        logging.warning(
            "%d images failed.  Check the log for details.  "
            "Re-run the script to skip already-cached files and retry failures.",
            len(errors),
        )

    log_gpu_memory("final")


if __name__ == "__main__":
    main()
