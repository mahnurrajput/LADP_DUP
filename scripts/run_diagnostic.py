"""
run_diagnostic.py — LADP Stage 1 End-to-End Diagnostic
=======================================================
Runs LLaVA-1.5 on the VizWiz val *tuning* split (first ``data.vizwiz_tuning_n``
examples, never index >= ``data.vizwiz_heldout_start``), computes the per-patch
reconstruction loss of g_phi on the image's own (E, T), the KNOR diagnostic, and
the correlation between loss and VQA error.

Consistency with the baseline (scripts/run_baseline.py)
--------------------------------------------------------
* Same prompt, greedy decoding settings and official VQA metric — all imported
  from ``src/vqa_eval.py`` (audit H1, H2).  ``vqa_acc`` is the official
  leave-one-out accuracy; ``any_match`` (>= 1 annotator agrees) is reported
  separately and never as "accuracy".
* E and T are captured at the projector (input / output), i.e. exactly the
  representation of the feature cache and of g_phi's training data (audit C1;
  ``src/feature_capture.py``).
* Answers are read from VizWiz's ``{"answer": ...}`` dicts (audit C2).
* g_phi and its ``norm_stats.pt`` must belong together (fingerprint; audit M4).
* Config paths resolve against the project root, not the working directory
  (audit M3).  Paths given on the command line are relative to the shell.

Usage
-----
  python scripts/run_diagnostic.py --config configs/pilot_mlp3_small.yaml

Outputs (in the config's output_dir)
------------------------------------
  diagnostic_results.json   — per-example records
  knor_results.json         — KNOR summary
  correlation_results.json  — loss vs VQA error (primary) and vs any_match (secondary)
  loss_vs_correctness.png   — mean loss by any_match label
  knor_distribution.png     — histogram of per-image KNOR values
  diagnostic_summary.txt    — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.config_paths import project_root_for, resolve_config_paths, resolve_path  # noqa: E402
from src.feature_capture import LlavaFeatureCapture  # noqa: E402
from src.vqa_eval import GENERATION, answer_strings, any_match, build_prompt, vqa_acc  # noqa: E402

logger = logging.getLogger("run_diagnostic")


# ---------------------------------------------------------------------------
# CLI and paths
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LADP Stage 1 Diagnostic: loss vs VQA error on the VizWiz val tuning split.")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "base_llava_vizwiz.yaml"),
                   help="Path to LADP YAML config.")
    p.add_argument("--recon_checkpoint", type=str, default=None,
                   help="g_phi state dict. Default: <checkpoints_dir>/<training.run_name>/best.pt")
    p.add_argument("--norm_stats", type=str, default=None,
                   help="NormStats file. Default: <checkpoints_dir>/<training.run_name>/norm_stats.pt")
    p.add_argument("--output_dir", type=str, default=None, help="Override output_dir from config.")
    p.add_argument("--n_examples", type=int, default=None,
                   help="Number of VizWiz examples (default: data.vizwiz_tuning_n).")
    p.add_argument("--knor_k", type=int, default=10, help="Neighbourhood size k for KNOR.")
    p.add_argument("--device", type=str, default=None, help="Override device (e.g. 'cuda', 'cpu').")
    p.add_argument("--require_norm_stats_fingerprint", action="store_true",
                   help="Refuse NormStats files without a fingerprint (e.g. pre-M4 files).")
    return p.parse_args(argv)


def resolve_run_paths(args: argparse.Namespace, cfg: dict, config_path: str) -> Dict[str, str]:
    """All file locations for one run, absolute.

    Config values resolve against the project root; command-line overrides
    against the current working directory.
    """
    root = project_root_for(config_path)
    rc = resolve_config_paths(cfg, root)
    data = rc.get("data", {})
    ckpt_dir = os.path.join(rc.get("checkpoints_dir") or str(root / "checkpoints"),
                            str(cfg.get("training", {}).get("run_name", "recon")))
    cli = lambda v: str(Path(v).resolve()) if v else None  # noqa: E731
    return {
        "root": str(root),
        "output_dir": cli(args.output_dir) or rc.get("output_dir") or str(root / "results" / "stage1_diagnostic"),
        "recon_checkpoint": cli(args.recon_checkpoint) or os.path.join(ckpt_dir, "best.pt"),
        "norm_stats": cli(args.norm_stats) or os.path.join(ckpt_dir, "norm_stats.pt"),
        "vizwiz_questions": data.get("vizwiz_questions") or resolve_path("data/vizwiz/val.json", root),
        "vizwiz_images": data.get("vizwiz_images") or resolve_path("data/vizwiz/val", root),
        "norm_calibration": (rc.get("loss") or {}).get("norm_calibration"),
    }


def check_heldout(n_examples: int, heldout_start: int) -> None:
    if n_examples > heldout_start:
        raise SystemExit(f"n_examples={n_examples} would read held-out VizWiz examples "
                         f"(index >= {heldout_start}); use at most {heldout_start}.")


# ---------------------------------------------------------------------------
# Data and scoring
# ---------------------------------------------------------------------------

def load_vizwiz_annotations(ann_path: str, n: int) -> List[dict]:
    """First ``n`` entries of the VizWiz val annotation list (``[{"image", "question", "answers"}]``)."""
    with open(ann_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("data", "annotations", "questions"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{ann_path}: expected a list of examples, got {type(data).__name__}")
    logger.info("VizWiz annotations: %d entries; using the first %d (tuning split).", len(data), min(n, len(data)))
    return data[:n]


def score_answer(pred: str, answers) -> Dict[str, object]:
    """Official VQA accuracy plus the separate any-match label for one prediction."""
    gts = answer_strings(answers)
    return {"vqa_acc": vqa_acc(pred, gts), "any_match": any_match(pred, gts), "ground_truth": gts}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_llava(model_name: str, device: torch.device, precision: str = "fp16"):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    if precision not in ("fp16", "fp32"):
        raise ValueError(f"model.precision must be 'fp16' or 'fp32', got {precision!r}")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    logger.info("Loading LLaVA-1.5 '%s' on %s (%s) …", model_name, device, dtype)
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return model, processor


def llava_answer(model, processor, image, question: str, device: torch.device) -> str:
    """One VQA query, exactly as in run_baseline.py (prompt, dtype, greedy decoding)."""
    inp = processor(images=image, text=build_prompt(question), return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inp, **GENERATION)
    return processor.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_loss_estimator(recon_ckpt: str, norm_stats_path: str, cfg: dict, device: torch.device,
                        require_fingerprint: bool = False, norm_calibration: Optional[str] = None):
    """g_phi + its NormStats (verified to belong together) -> LossEstimator."""
    from loss_estimator import (LossEstimator, NormStats, apply_norm_calibration,
                                build_recon_net_from_config, check_norm_stats_pairing)

    ns = NormStats.load(norm_stats_path)
    net = build_recon_net_from_config(cfg, d_in=ns.T_mean.shape[0], d_out=ns.E_mean.shape[0])
    net.load_state_dict(torch.load(recon_ckpt, map_location="cpu", weights_only=True))
    check_norm_stats_pairing(ns, net, require_meta=require_fingerprint)
    ns = apply_norm_calibration(ns, norm_calibration)
    net.to(device).eval()
    distance = cfg.get("loss", {}).get("distance", "mse")
    logger.info("g_phi %s (%s params) | distance=%s | p5=%.4f p95=%.4f", net.depth,
                f"{net.param_count():,}", distance, ns.L_p5, ns.L_p95)
    return LossEstimator(net, ns, distance=distance)


# ---------------------------------------------------------------------------
# KNOR, plots, summary
# ---------------------------------------------------------------------------

def compute_knor(E_pooled_list: List[torch.Tensor], T_pooled_list: List[torch.Tensor], k: int) -> dict:
    from diagnostics.knor import KNORDiagnostic

    results = KNORDiagnostic(k=k).compute(torch.stack(E_pooled_list), torch.stack(T_pooled_list))
    results["knor_per_image_list"] = results.pop("knor_per_image").tolist()
    logger.info("KNOR (k=%d): mean=%.4f, std=%.4f", results["k"], results["knor_mean"], results["knor_std"])
    return results


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return None


def plot_loss_vs_correctness(records: List[dict], out_path: str) -> None:
    """Box plot of mean loss for any_match = True vs False."""
    plt = _plt()
    if plt is None:
        return
    import random
    groups = [[r["mean_loss"] for r in records if r["any_match"]],
              [r["mean_loss"] for r in records if not r["any_match"]]]
    fig, ax = plt.subplots(figsize=(6, 5))
    # Tick labels set separately: boxplot(labels=) was renamed tick_labels in
    # matplotlib 3.9 and removed in 3.11.
    ax.boxplot(groups, patch_artist=True,
               boxprops=dict(facecolor="#a8d8a8", color="#333"), medianprops=dict(color="#c0392b", linewidth=2))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["any annotator agrees", "no annotator agrees"])
    rng = random.Random(42)
    for i, g in enumerate(groups, start=1):
        s = rng.sample(g, min(200, len(g)))
        ax.scatter([i + rng.uniform(-0.07, 0.07) for _ in s], s, alpha=0.25, s=8, color="#2c3e50", zorder=3)
    ax.set_ylabel("Mean patch reconstruction loss")
    ax.set_title("Loss vs answer agreement — VizWiz val (tuning split)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_knor_distribution(knor_per_image: List[float], knor_mean: float, k: int, out_path: str) -> None:
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(knor_per_image, bins=k + 1, range=(0, 1), color="#5b9bd5", edgecolor="white", linewidth=0.5)
    ax.axvline(knor_mean, color="#c0392b", linewidth=2, linestyle="--", label=f"Mean = {knor_mean:.3f}")
    ax.set_xlabel(f"KNOR (k={k})")
    ax.set_ylabel("Number of images")
    ax.set_title("KNOR Distribution — VizWiz val (tuning split)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_summary(records: List[dict], corr_vqa: dict, corr_match: dict, knor: dict, elapsed: float) -> str:
    valid = [r for r in records if r["mean_loss"] == r["mean_loss"]]
    n = len(records)
    acc = sum(r["vqa_acc"] for r in records) / max(n, 1)
    match = sum(bool(r["any_match"]) for r in records) / max(n, 1)
    f = lambda d, k: d.get(k, float("nan"))  # noqa: E731
    return "\n".join([
        "=" * 65, "LADP Stage 1 Diagnostic — Summary Report", "=" * 65, "",
        f"  Examples evaluated      : {n}  (VizWiz val tuning split)",
        f"  VQA accuracy (official) : {acc:.4f}",
        f"  Any-annotator match     : {100 * match:.1f}%  (binary label; not VQA accuracy)",
        f"  Mean patch loss         : {sum(r['mean_loss'] for r in valid) / max(len(valid), 1):.5f}"
        f"  ({len(valid)} examples with features)",
        "",
        "--- Loss vs VQA error (primary: error = 1 - VQA accuracy) ---",
        f"  Pearson  r              : {f(corr_vqa, 'pearson_r'):+.4f}  (p={f(corr_vqa, 'pearson_p'):.4g})",
        f"  Spearman r              : {f(corr_vqa, 'spearman_r'):+.4f}  (p={f(corr_vqa, 'spearman_p'):.4g})",
        "  (positive r: higher loss goes with lower accuracy)",
        "",
        "--- Loss vs no-annotator-match (secondary, binary) ---",
        f"  Pearson  r              : {f(corr_match, 'pearson_r'):+.4f}  (p={f(corr_match, 'pearson_p'):.4g})",
        f"  Mean loss (match / none): {f(corr_match, 'mean_loss_correct'):.5f} / {f(corr_match, 'mean_loss_wrong'):.5f}",
        "",
        "--- KNOR ---",
        f"  k / mean / std          : {knor.get('k')} / {f(knor, 'knor_mean'):.4f} / {f(knor, 'knor_std'):.4f}",
        "  (1.0 = neighbourhoods fully preserved by the connector; no threshold is taken from the paper)",
        "",
        f"  Elapsed time            : {elapsed:.1f}s", "=" * 65,
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    paths = resolve_run_paths(args, cfg, args.config)

    data_cfg = cfg.get("data", {})
    n_examples = args.n_examples or int(data_cfg.get("vizwiz_tuning_n", 1000))
    check_heldout(n_examples, int(data_cfg.get("vizwiz_heldout_start", 3800)))
    device = torch.device(args.device or cfg.get("model", {}).get("device", "cpu"))
    os.makedirs(paths["output_dir"], exist_ok=True)
    for k, v in paths.items():
        logger.info("%-17s: %s", k, v)

    annotations = load_vizwiz_annotations(paths["vizwiz_questions"], n_examples)
    estimator = load_loss_estimator(paths["recon_checkpoint"], paths["norm_stats"], cfg, device,
                                    require_fingerprint=args.require_norm_stats_fingerprint,
                                    norm_calibration=paths["norm_calibration"])
    from diagnostics.knor import correlate_loss_with_error, correlate_loss_with_vqa

    model_cfg = cfg.get("model", {})
    model, processor = load_llava(model_cfg.get("name", "llava-hf/llava-1.5-7b-hf"), device,
                                  model_cfg.get("precision", "fp16"))
    norm_mode = cfg.get("loss", {}).get("norm", "robust")

    from PIL import Image

    records: List[dict] = []
    E_pooled, T_pooled = [], []
    t_start = time.time()
    for idx, ann in enumerate(annotations):
        image_path = os.path.join(paths["vizwiz_images"], ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Cannot open image %s: %s — skipping.", image_path, exc)
            continue

        with LlavaFeatureCapture(model, to_float=True) as cap:
            pred = llava_answer(model, processor, image, ann["question"], device)

        mean_loss = mean_loss_norm = float("nan")
        if cap.E is not None and cap.T is not None:
            if cap.E.shape[0] != cap.T.shape[0]:
                raise RuntimeError(f"E/T patch counts differ: {tuple(cap.E.shape)} vs {tuple(cap.T.shape)}")
            L = estimator.compute(cap.E, cap.T)
            mean_loss = float(L.mean())
            mean_loss_norm = float(_normalise(L, estimator.norm_stats, norm_mode).mean())
            E_pooled.append(cap.E.mean(0))
            T_pooled.append(cap.T.mean(0))
        else:
            logger.warning("Example %d (%s): features not captured — loss = NaN.", idx, ann["image"])

        s = score_answer(pred, ann.get("answers", []))
        records.append({"idx": idx, "image_id": ann["image"], "question": ann["question"],
                        "llava_answer": pred, "ground_truth": s["ground_truth"],
                        "vqa_acc": s["vqa_acc"], "any_match": s["any_match"],
                        "mean_loss": mean_loss, "mean_loss_norm": mean_loss_norm})
        if (idx + 1) % 50 == 0:
            valid = [r["mean_loss"] for r in records if r["mean_loss"] == r["mean_loss"]]
            logger.info("Progress: %d/%d | VQA acc=%.4f | mean_loss=%.4f", idx + 1, n_examples,
                        sum(r["vqa_acc"] for r in records) / len(records),
                        sum(valid) / max(len(valid), 1))

    elapsed = time.time() - t_start
    out = paths["output_dir"]
    with open(os.path.join(out, "diagnostic_results.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    knor: dict = {"knor_mean": float("nan"), "knor_std": float("nan"), "k": args.knor_k}
    if len(E_pooled) >= args.knor_k + 1:
        knor = compute_knor(E_pooled, T_pooled, k=args.knor_k)
        with open(os.path.join(out, "knor_results.json"), "w", encoding="utf-8") as fh:
            json.dump(knor, fh, indent=2)
    else:
        logger.warning("Only %d examples with features; KNOR (k=%d) skipped.", len(E_pooled), args.knor_k)

    valid = [r for r in records if r["mean_loss"] == r["mean_loss"]]
    corr_vqa: dict = {}
    corr_match: dict = {}
    if len(valid) >= 3:
        losses = [r["mean_loss"] for r in valid]
        corr_vqa = correlate_loss_with_vqa(losses, [r["vqa_acc"] for r in valid])
        corr_match = correlate_loss_with_error(losses, [bool(r["any_match"]) for r in valid])
        with open(os.path.join(out, "correlation_results.json"), "w", encoding="utf-8") as fh:
            json.dump({"primary_loss_vs_vqa_error": corr_vqa,
                       "secondary_loss_vs_no_annotator_match": corr_match}, fh, indent=2)
    if valid:
        plot_loss_vs_correctness(valid, os.path.join(out, "loss_vs_correctness.png"))
    if "knor_per_image_list" in knor:
        plot_knor_distribution(knor["knor_per_image_list"], knor["knor_mean"], knor["k"],
                               os.path.join(out, "knor_distribution.png"))

    summary = build_summary(records, corr_vqa, corr_match, knor, elapsed)
    with open(os.path.join(out, "diagnostic_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print("\n" + summary)


def _normalise(L: torch.Tensor, ns, mode: str) -> torch.Tensor:
    """Normalised loss with the shared implementation (src/normaliser.py)."""
    from src.normaliser import Normaliser
    return Normaliser().normalise_loss(L, ns, mode=mode)


if __name__ == "__main__":
    main()
