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
  <cache_dir>/<split>/verified_index.json   (size/mtime of files already verified)

Each .pt file contains:
  {
      "E":          Tensor[N, 1024],   # pre-connector CLIP patch features (CPU)
      "T":          Tensor[N, 4096],   # post-connector projected tokens   (CPU)
      "image_path": str,               # absolute path of the source image
      "image_idx":  int,               # index i (zero-based) in this split
  }
File i always holds image i of the *sorted* image list, so the existing
10,000-image cache is exactly files 0–9,999 of the full COCO cache.

Resumable and idempotent
------------------------
  * Existing files are verified and reused; only missing/invalid ones are
    (re)extracted.  Verification ("quick", default) memory-maps the file and
    checks image name, index, shapes and dtype without reading the tensors;
    "full" also reads them and checks they are finite.  Files verified on an
    earlier run are skipped if their size and mtime are unchanged
    (verified_index.json); --reverify forces a fresh check.
  * Every file is written to <name>.tmp and renamed (atomic); stale .tmp
    files from a crash are removed.
  * Before starting and every --space_check_every images, free disk space is
    checked; the run stops cleanly if it would drop below --min_free_gb.
  * Progress, rate and ETA are logged every --log_every images; the manifest
    is rewritten (atomically) every --manifest_every images.

Failure policy and completion (audit M5)
----------------------------------------
  * Every image has a status in manifest.json: cached, failed (with error and
    attempt count, accumulated across runs), excluded (with reason), or
    pending (not attempted yet).  "counts" summarises them.
  * Failed images are retried on every run.  A permanently unreadable image is
    excluded only explicitly: --exclude_failed after --min_attempts (default 2)
    failed attempts.  Excluded images are not retried unless --retry_excluded.
  * "complete" is true only if the run was not stopped early and every image is
    cached or explicitly excluded (see COMPLETION_CRITERIA).  The trainer
    (data.require_complete_cache) accepts exclusions only when they are listed.
  * Exit codes: 2 not enough disk space at start, 3 stopped early (disk space),
    4 finished but incomplete (failed images remain).
"""

import argparse
import json
import logging
import os
import shutil
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

EXPECTED_SHAPES = {"E": (576, 1024), "T": (576, 4096)}   # LLaVA-1.5 @ 336 px
EXPECTED_DTYPE = torch.float16


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
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
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
# Verification, atomic writes, disk space
# ---------------------------------------------------------------------------

def verify_cache_file(path: Path, index: int, image_path: Path, level: str = "quick") -> str | None:
    """Return None if ``path`` is a valid cache file for image ``index``,
    otherwise a short reason string.

    quick : memory-maps the file (tensor bytes are not read) and checks keys,
            image file name, index, shapes and dtype — catches truncated or
            foreign files cheaply (a truncated zip fails to open).
    full  : additionally reads both tensors and checks they are finite.
    """
    try:
        d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:  # noqa: BLE001
        return f"unreadable ({type(exc).__name__}: {exc})"
    for k in ("E", "T", "image_path", "image_idx"):
        if k not in d:
            return f"missing key {k!r}"
    if Path(str(d["image_path"])).name != image_path.name:
        return f"image mismatch (file has {Path(str(d['image_path'])).name}, expected {image_path.name})"
    if int(d["image_idx"]) != index:
        return f"index mismatch ({d['image_idx']} != {index})"
    for k, shape in EXPECTED_SHAPES.items():
        if tuple(d[k].shape) != shape:
            return f"{k} shape {tuple(d[k].shape)} != {shape}"
        if d[k].dtype != EXPECTED_DTYPE:
            return f"{k} dtype {d[k].dtype} != {EXPECTED_DTYPE}"
    if level == "full":
        for k in ("E", "T"):
            if not torch.isfinite(d[k].float()).all():
                return f"{k} has non-finite values"
    return None


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


class _CacheImages(torch.utils.data.Dataset):
    """(index, pixel_values | None, error | None) for the images to extract."""

    def __init__(self, items: list[tuple[int, str]], image_processor) -> None:
        self.items = items
        self.proc = image_processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, j: int):
        i, p = self.items[j]
        try:
            with Image.open(p) as im:
                px = self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            return i, px, None
        except Exception as exc:  # noqa: BLE001
            return i, None, f"{type(exc).__name__}: {exc}"


def _collate(batch):
    ok = [(i, px) for i, px, err in batch if err is None]
    bad = [(i, err) for i, px, err in batch if err is not None]
    if ok:
        idx = [i for i, _ in ok]
        return idx, torch.stack([px for _, px in ok]), bad
    return [], None, bad


# ---------------------------------------------------------------------------
# Core caching logic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Failure policy and completion criteria (audit M5)
# ---------------------------------------------------------------------------

COMPLETION_CRITERIA = (
    "complete = the run was not stopped early AND every image is either cached "
    "(file present and verified) or explicitly excluded (listed in 'excluded' with a "
    "reason) AND no image is failed or pending. Failed images are retried on every "
    "run; an image is excluded only with --exclude_failed after at least "
    "--min_attempts failed attempts, and stays excluded (not retried) unless "
    "--retry_excluded is given.")


def load_previous_manifest(path: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    """(failed, excluded) records of an earlier run, keyed by image index."""
    if not path.exists():
        return {}, {}
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logging.warning("Previous manifest %s unreadable; starting the failure history afresh.", path)
        return {}, {}
    failed = {int(r["index"]): r for r in m.get("failed", []) if "index" in r}
    excluded = {int(r["index"]): r for r in m.get("excluded", []) if "index" in r}
    return failed, excluded


def build_manifest(image_paths: list[Path], entries: list[dict], run_errors: list[dict],
                   prev_failed: dict[int, dict], prev_excluded: dict[int, dict],
                   stopped_early: bool, exclude_failed: bool = False, min_attempts: int = 2,
                   retry_excluded: bool = False, meta: dict | None = None) -> dict:
    """Manifest with an explicit status for every image.

    cached    : entry in ``entries`` (file written or verified this run)
    failed    : failed this run (attempts = earlier attempts + 1), or failed
                earlier and not retried yet; retried on the next run
    excluded  : failed >= min_attempts times and --exclude_failed given now or
                earlier (kept unless --retry_excluded); never counted as cached
    pending   : none of the above (not attempted yet, e.g. stopped for disk space)
    """
    if min_attempts < 1:
        raise ValueError("min_attempts must be >= 1")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    cached = {int(e["index"]) for e in entries}
    failed: dict[int, dict] = {}
    for i, r in prev_failed.items():                       # earlier failures not resolved yet
        if i not in cached:
            failed[i] = dict(r)
    for e in run_errors:                                   # this run's failures
        i = int(e["index"])
        attempts = int(prev_failed.get(i, {}).get("attempts", prev_excluded.get(i, {}).get("attempts", 0))) + 1
        failed[i] = {"index": i, "image_path": str(e["image_path"]), "error": str(e["error"]),
                     "attempts": attempts, "last_attempt": now}
    excluded: dict[int, dict] = {}
    if not retry_excluded:
        excluded = {i: dict(r) for i, r in prev_excluded.items() if i not in cached and i not in failed}
    if exclude_failed:
        for i in [i for i, r in failed.items() if int(r.get("attempts", 0)) >= min_attempts]:
            r = failed.pop(i)
            excluded[i] = {**r, "excluded_at": now,
                           "reason": f"failed {r['attempts']} attempt(s); last error: {r['error']}"}
    retried_ok = sorted(i for i in prev_failed if i in cached)
    n = len(image_paths)
    pending = [i for i in range(n) if i not in cached and i not in failed and i not in excluded]
    complete = (not stopped_early) and not failed and not pending and len(cached) + len(excluded) == n
    counts = {"total": n, "cached": len(cached), "failed": len(failed), "excluded": len(excluded),
              "pending": len(pending), "retried_ok_this_run": len(retried_ok)}
    return {**(meta or {}),
            "total_images": n, "cached_ok": len(cached), "errors": len(failed),
            "complete": bool(complete), "stopped_early": bool(stopped_early),
            "counts": counts, "completion_criteria": COMPLETION_CRITERIA,
            "files": sorted(entries, key=lambda e: e["index"]),
            "failed": [failed[i] for i in sorted(failed)],
            "excluded": [excluded[i] for i in sorted(excluded)],
            "pending_first": pending[:50]}


def cache_split(
    extractor,
    image_paths: list[Path],
    out_dir: Path,
    engine: str = "vision_projector",
    batch_size: int = 32,
    num_workers: int = 8,
    verify_level: str = "quick",
    reverify: bool = False,
    min_free_gb: float = 40.0,
    space_check_every: int = 256,
    log_every: int = 1000,
    manifest_every: int = 2000,
    manifest_writer=None,
    skip_indices: set[int] | None = None,
) -> tuple[list[dict], list[dict], bool]:
    """Verify existing files, then extract only what is missing.

    Returns (manifest_entries, errors, completed).  ``completed`` is False if
    the run stopped early because free space fell below ``min_free_gb``.
    ``skip_indices`` (explicitly excluded images) are not extracted when their
    file is missing; a valid existing file for them is still reused.
    """
    skip_indices = skip_indices or set()
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []
    entries: dict[int, dict] = {}

    # -- remove stale temp files from an interrupted run ----------------------
    for tmp in out_dir.glob("features_*.pt.tmp"):
        logging.warning("Removing stale temp file from an interrupted run: %s", tmp.name)
        tmp.unlink()

    # -- 1. verify what is already there --------------------------------------
    vidx_path = out_dir / "verified_index.json"
    vidx: dict = {}
    if vidx_path.exists() and not reverify:
        try:
            vidx = json.loads(vidx_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logging.warning("verified_index.json unreadable — re-verifying all files")
    todo: list[tuple[int, str]] = []
    n_reused = n_trusted = n_bad = 0
    t0 = time.time()
    for i, img_path in enumerate(tqdm(image_paths, desc="Verifying existing cache", unit="file",
                                      dynamic_ncols=True, mininterval=2.0)):
        f = out_dir / f"features_{i:06d}.pt"
        if not f.exists():
            if i not in skip_indices:
                todo.append((i, str(img_path)))
            continue
        st = f.stat()
        key = f.name
        rec = vidx.get(key)
        if (rec and rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime_ns
                and rec.get("image") == img_path.name and rec.get("level") in ("quick", "full")
                and (verify_level == "quick" or rec.get("level") == "full")):
            n_trusted += 1
        else:
            reason = verify_cache_file(f, i, img_path, verify_level)
            if reason is not None:
                logging.warning("Invalid cache file %s (%s) — will re-extract", f.name, reason)
                n_bad += 1
                todo.append((i, str(img_path)))
                continue
            vidx[key] = {"size": st.st_size, "mtime": st.st_mtime_ns,
                         "image": img_path.name, "level": verify_level}
        n_reused += 1
        entries[i] = {"index": i, "filename": f.name, "image_path": str(img_path),
                      "E_shape": list(EXPECTED_SHAPES["E"]), "T_shape": list(EXPECTED_SHAPES["T"]),
                      "cached": True}
    atomic_json_dump(vidx, vidx_path)
    logging.info("Existing cache: %d valid (%d trusted from verified_index.json), "
                 "%d invalid, %d missing → %d to extract  [%.0f s]",
                 n_reused, n_trusted, n_bad, len(todo) - n_bad, len(todo), time.time() - t0)

    if not todo:
        return [entries[i] for i in sorted(entries)], errors, True

    # -- 2. disk space check before starting ----------------------------------
    sample = next(iter(out_dir.glob("features_*.pt")), None)
    per_file = sample.stat().st_size if sample else 5_900_261
    need_gb = len(todo) * per_file / 1e9
    have_gb = free_gb(out_dir)
    logging.info("Disk: %.1f GB free; %d files x %.2f MB = %.1f GB needed; safety margin %.0f GB",
                 have_gb, len(todo), per_file / 1e6, need_gb, min_free_gb)
    if have_gb - need_gb < min_free_gb:
        short = need_gb + min_free_gb - have_gb
        logging.error("NOT ENOUGH DISK SPACE: free %.1f GB, need %.1f GB + %.0f GB margin. "
                      "Free at least %.1f GB more on %s, then re-run (finished files are kept).",
                      have_gb, need_gb, min_free_gb, short, out_dir.anchor)
        raise SystemExit(2)

    # -- 3. extract ------------------------------------------------------------
    log_gpu_memory("before extraction loop")
    completed = True
    n_done, t_start = 0, time.time()

    def record(i: int, E: torch.Tensor, T: torch.Tensor) -> None:
        img = image_paths[i]
        f = out_dir / f"features_{i:06d}.pt"
        atomic_torch_save({"E": E, "T": T, "image_path": str(img), "image_idx": i}, f)
        st = f.stat()
        vidx[f.name] = {"size": st.st_size, "mtime": st.st_mtime_ns,
                        "image": img.name, "level": "written"}
        entries[i] = {"index": i, "filename": f.name, "image_path": str(img),
                      "E_shape": list(E.shape), "T_shape": list(T.shape), "cached": False}

    def progress(force: bool = False) -> None:
        if force or (n_done and n_done % log_every < batch_size):
            el = time.time() - t_start
            rate = n_done / max(el, 1e-6)
            eta = (len(todo) - n_done) / max(rate, 1e-9)
            logging.info("Progress: %d / %d extracted (%.1f img/s, %d errors) | elapsed %.1f min | "
                         "ETA %.1f h | free %.1f GB", n_done, len(todo), rate, len(errors),
                         el / 60, eta / 3600, free_gb(out_dir))

    if engine == "adapter":
        # Original single-image path through the full LLaVA model.
        for i, p in tqdm(todo, desc="Caching features", unit="img", dynamic_ncols=True):
            if n_done % space_check_every == 0 and free_gb(out_dir) < min_free_gb:
                completed = False
                break
            try:
                feats = extractor.extract_features(Image.open(p))
                record(i, feats["E"].cpu(), feats["T"].cpu())
            except Exception as exc:  # noqa: BLE001
                logging.error("Failed image %d (%s): %s", i, p, exc)
                errors.append({"index": i, "image_path": p, "error": str(exc)})
            n_done += 1
            progress()
            if manifest_writer and n_done % manifest_every == 0:
                manifest_writer([entries[k] for k in sorted(entries)], errors, vidx)
    else:
        ds = _CacheImages(todo, extractor.image_processor)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                                         collate_fn=_collate, pin_memory=True)
        since_check = 0
        for idx, px, bad in tqdm(dl, desc="Caching features", unit="batch", dynamic_ncols=True):
            for i, err in bad:
                logging.error("Failed image %d (%s): %s", i, image_paths[i], err)
                errors.append({"index": i, "image_path": str(image_paths[i]), "error": err})
            if idx:
                E, T = extractor(px)
                E, T = E.cpu(), T.cpu()
                for j, i in enumerate(idx):
                    record(i, E[j].clone(), T[j].clone())
            n_done += len(idx) + len(bad)
            since_check += len(idx) + len(bad)
            progress()
            if manifest_writer and n_done % manifest_every < batch_size:
                manifest_writer([entries[k] for k in sorted(entries)], errors, vidx)
            if since_check >= space_check_every:
                since_check = 0
                if free_gb(out_dir) < min_free_gb:
                    completed = False
                    break

    if not completed:
        logging.error("STOPPED: free space on %s fell below %.0f GB. %d files written this run "
                      "are kept; free space and re-run to continue.",
                      out_dir.anchor, min_free_gb, n_done)
    progress(force=True)
    atomic_json_dump(vidx, vidx_path)
    if errors:
        logging.warning("%d image(s) failed and were skipped.  Error details above.", len(errors))
    return [entries[i] for i in sorted(entries)], errors, completed


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
        help="Limit processing to this many images (first N, sorted).  For "
             "coco_train the default is data.max_images from the config.",
    )
    parser.add_argument("--engine", choices=["vision_projector", "adapter"], default="vision_projector",
                        help="vision_projector: CLIP tower + projector only, batched (default; "
                             "bit-identical to the adapter on checked images).  adapter: original "
                             "one-image-at-a-time path through the full LLaVA model.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="images per GPU batch (vision_projector).  Default 1: bit-identical to "
                             "the existing cache and to LLaVA inference.  Larger batches are faster "
                             "but change ~0.3%% of patches noticeably (cos < 0.99, measured on 32 "
                             "images) because of fp16 kernel differences.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers for JPEG decode")
    parser.add_argument("--verify", choices=["quick", "full"], default="quick",
                        help="integrity check for existing files (see module docstring)")
    parser.add_argument("--reverify", action="store_true",
                        help="ignore verified_index.json and re-check every existing file")
    parser.add_argument("--min_free_gb", type=float, default=40.0,
                        help="stop if free space would drop below this many GB (default 40)")
    parser.add_argument("--space_check_every", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--manifest_every", type=int, default=2000)
    parser.add_argument("--hf_home", default=os.environ.get("HF_HOME"),
                        help="Hugging Face cache root (vision_projector engine)")
    parser.add_argument("--exclude_failed", action="store_true",
                        help="explicitly exclude images that have failed at least --min_attempts times "
                             "(recorded in the manifest with the error); without it they stay 'failed' "
                             "and the cache is never complete")
    parser.add_argument("--min_attempts", type=int, default=2,
                        help="failed attempts (across runs) before --exclude_failed may exclude an image")
    parser.add_argument("--retry_excluded", action="store_true",
                        help="retry previously excluded images instead of keeping them excluded")
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

    # For the COCO split, default the image count to data.max_images (the same
    # value the trainer uses) when --max_images is not given.
    if args.max_images is None and args.split == "coco_train":
        mi = data_cfg.get("max_images")
        args.max_images = None if mi in (None, "all") else int(mi)

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

    class _Lazy:
        """Load the model only if something actually needs extracting."""

        def __init__(self):
            self._m = None

        def _load(self):
            if self._m is None:
                logging.info("Loading %s: model=%s  precision=%s  device=%s",
                             args.engine, model_name, precision, device)
                t0_load = time.time()
                try:
                    if args.engine == "adapter":
                        from connector_adapter import LlavaAdapter
                        self._m = LlavaAdapter(model_name=model_name, device=device, precision=precision)
                    else:
                        from onthefly_features import VisionProjector
                        self._m = VisionProjector(
                            model_name, hf_home=args.hf_home, device=device,
                            dtype=torch.float16 if precision == "fp16" else torch.float32)
                except Exception as exc:
                    logging.error("Failed to load the model: %s", exc, exc_info=True)
                    sys.exit(1)
                logging.info("Model loaded in %.1f s", time.time() - t0_load)
                log_gpu_memory("after model load")
            return self._m

        @property
        def image_processor(self):
            return self._load().image_processor

        def extract_features(self, image):
            return self._load().extract_features(image)

        def __call__(self, px):
            return self._load()(px)

    extractor = _Lazy()
    manifest_path = out_dir / "manifest.json"
    t0_extract = time.time()
    prev_failed, prev_excluded = load_previous_manifest(manifest_path)
    if prev_failed or prev_excluded:
        logging.info("Earlier runs: %d failed image(s) to retry, %d excluded%s", len(prev_failed),
                     len(prev_excluded), " (will be retried: --retry_excluded)" if args.retry_excluded else "")

    def manifest_for(entries, errs, stopped_early):
        return build_manifest(
            image_paths, entries, errs, prev_failed, prev_excluded, stopped_early,
            exclude_failed=args.exclude_failed, min_attempts=args.min_attempts,
            retry_excluded=args.retry_excluded,
            meta={"split": args.split, "image_dir": str(image_dir), "cache_dir": str(out_dir),
                  "engine": args.engine, "elapsed_s": round(time.time() - t0_extract, 1)})

    def write_manifest(entries, errs, _vidx=None):
        # periodic snapshot while running: never complete
        atomic_json_dump(manifest_for(entries, errs, stopped_early=True), manifest_path)

    # ------------------------------------------------------------------
    # Verify existing files, extract the rest.
    # ------------------------------------------------------------------
    entries, errors, completed = cache_split(
        extractor, image_paths, out_dir,
        engine=args.engine, batch_size=args.batch_size, num_workers=args.num_workers,
        verify_level=args.verify, reverify=args.reverify, min_free_gb=args.min_free_gb,
        space_check_every=args.space_check_every, log_every=args.log_every,
        manifest_every=args.manifest_every, manifest_writer=write_manifest,
        skip_indices=set() if args.retry_excluded else set(prev_excluded),
    )
    elapsed = time.time() - t0_extract
    manifest = manifest_for(entries, errors, stopped_early=not completed)
    atomic_json_dump(manifest, manifest_path)
    logging.info("Manifest written to %s", manifest_path)

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    c = manifest["counts"]
    logging.info("=" * 60)
    logging.info("  Total images   : %d", c["total"])
    logging.info("  Cached         : %d  (%d earlier failures recovered this run)", c["cached"],
                 c["retried_ok_this_run"])
    logging.info("  Failed         : %d  (retried automatically on the next run)", c["failed"])
    logging.info("  Excluded       : %d  (explicit, listed in the manifest)", c["excluded"])
    logging.info("  Pending        : %d", c["pending"])
    logging.info("  COMPLETE       : %s", manifest["complete"])
    logging.info("  Elapsed        : %.1f s", elapsed)
    for r in manifest["failed"][:20]:
        logging.warning("  failed #%d (%d attempt(s)): %s — %s", r["index"], r["attempts"],
                        Path(r["image_path"]).name, r["error"])
    if c["failed"]:
        logging.warning("%d image(s) failed. Re-run to retry them; if an image is permanently "
                        "unreadable, re-run with --exclude_failed (after %d attempts) to exclude it "
                        "explicitly.", c["failed"], args.min_attempts)

    log_gpu_memory("final")
    if not completed:
        sys.exit(3)
    if not manifest["complete"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
