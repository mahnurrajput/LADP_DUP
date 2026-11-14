"""
run_diagnostic.py — LADP Stage 1 End-to-End Diagnostic
=======================================================
Runs baseline LLaVA-1.5 inference on VizWiz val (tuning split, N=1000),
computes per-image reconstruction loss via LossEstimator, computes KNOR,
correlates loss with VQA error, and writes all outputs to output_dir.

Usage
-----
  python scripts/run_diagnostic.py \\
      --config configs/base_llava_vizwiz.yaml \\
      --recon_checkpoint checkpoints/recon/best.pt \\
      --norm_stats checkpoints/recon/norm_stats.pt

Outputs (in output_dir defined in the config, default: results/stage1_diagnostic/)
---------
  diagnostic_results.json   — per-example records
  knor_results.json         — KNOR summary
  correlation_results.json  — Pearson/Spearman r
  loss_vs_correctness.png   — box plot of mean loss for correct vs wrong
  knor_distribution.png     — histogram of per-image KNOR values
  diagnostic_summary.txt    — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LADP Stage 1 Diagnostic: loss vs VQA correctness on VizWiz val."
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/base_llava_vizwiz.yaml",
        help="Path to LADP YAML config.",
    )
    p.add_argument(
        "--recon_checkpoint",
        type=str,
        default="checkpoints/recon/best.pt",
        help="Path to trained gφ state dict (.pt).",
    )
    p.add_argument(
        "--norm_stats",
        type=str,
        default="checkpoints/recon/norm_stats.pt",
        help="Path to NormStats checkpoint (.pt).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output_dir from config.",
    )
    p.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Override number of VizWiz examples (default from config).",
    )
    p.add_argument(
        "--knor_k",
        type=int,
        default=10,
        help="Neighbourhood size k for KNOR (default: 10).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cuda', 'cpu').",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# VizWiz annotation helpers
# ---------------------------------------------------------------------------

def load_vizwiz_annotations(ann_path: str, n: int) -> List[dict]:
    """Load the first *n* entries from the VizWiz val annotation JSON.

    The JSON is a list of dicts with keys:
        "image"    — filename, e.g. "VizWiz_val_00000001.jpg"
        "question" — question string
        "answers"  — list of up to 10 answer strings
    """
    with open(ann_path, "r") as fh:
        data = json.load(fh)

    # Accept both a bare list and {"data": [...]} style.
    if isinstance(data, dict):
        # Try common wrapper keys.
        for key in ("data", "annotations", "questions"):
            if key in data:
                data = data[key]
                break

    logger.info("VizWiz annotations: %d total entries found.", len(data))
    subset = data[:n]
    logger.info("Using first %d examples (tuning split).", len(subset))
    return subset


# ---------------------------------------------------------------------------
# VizWiz VQA accuracy
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))


def _normalise_answer(ans: str) -> str:
    """Lower-case and strip punctuation (standard VQA normalisation)."""
    return _PUNCT_RE.sub("", ans.lower()).strip()


def vizwiz_accuracy(predicted: str, ground_truth_answers: List[str]) -> float:
    """VizWiz VQA accuracy for a single example.

    Formula: acc = min(count_matching / 3, 1.0)
    where count_matching = number of GT answers (out of 10) that match the
    predicted answer after normalisation.
    """
    pred_norm = _normalise_answer(predicted)
    count_matching = sum(
        1 for gt in ground_truth_answers if _normalise_answer(gt) == pred_norm
    )
    return min(count_matching / 3, 1.0)


def is_correct(predicted: str, ground_truth_answers: List[str]) -> bool:
    """Return True if vizwiz_accuracy >= 1/3 (i.e. at least 1 match)."""
    return vizwiz_accuracy(predicted, ground_truth_answers) > 0.0


# ---------------------------------------------------------------------------
# LLaVA-1.5 inference helpers
# ---------------------------------------------------------------------------

def load_llava(model_name: str, device: torch.device, precision: str = "fp16"):
    """Load LLaVA-1.5 model and processor from HuggingFace."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    dtype = torch.float16 if precision == "fp16" else torch.float32
    logger.info("Loading LLaVA-1.5 model '%s' on %s (dtype=%s) …", model_name, device, dtype)

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    logger.info("LLaVA-1.5 model loaded (%d parameters).", sum(p.numel() for p in model.parameters()))
    return model, processor


def llava_answer(
    model,
    processor,
    image,
    question: str,
    device: torch.device,
) -> str:
    """Run a single VQA query through LLaVA-1.5 and return the answer string."""
    prompt = (
        f"USER: <image>\n{question} "
        "Answer the question using a single word or phrase. ASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    full_text = processor.decode(output_ids[0], skip_special_tokens=True)
    answer = full_text.split("ASSISTANT:")[-1].strip()
    return answer


# ---------------------------------------------------------------------------
# Per-patch feature extraction hook
# ---------------------------------------------------------------------------

class _FeatureCapture:
    """Context manager that hooks into LLaVA-1.5 to capture:
        E  — pre-connector CLIP vision features   (N, dv)
        T  — post-connector LLM tokens            (N, dt)
    """

    def __init__(self, model) -> None:
        self._model = model
        self._hooks: List[Any] = []
        self.E: Optional[torch.Tensor] = None   # (N, dv)
        self.T: Optional[torch.Tensor] = None   # (N, dt)

    def __enter__(self):
        self._register()
        return self

    def __exit__(self, *_):
        self._remove()

    def _register(self) -> None:
        """Register forward hooks on the CLIP vision tower output and
        the MLP projector output (post-connector tokens)."""

        # LLaVA-1.5 (HuggingFace): model.vision_tower outputs the CLIP features.
        # The connector (mm_projector / multi_modal_projector) maps them to LLM space.
        # We capture:
        #   E — output of the CLIP vision tower (the penultimate hidden states)
        #   T — output of the multi-modal projector (post-connector)

        def _hook_vision_tower(module, inp, out):
            # out may be a BaseModelOutputWithPooling or a plain tensor
            if hasattr(out, "last_hidden_state"):
                # Remove [CLS] token (index 0) → (N, dv)
                self.E = out.last_hidden_state[:, 1:, :].squeeze(0).detach().cpu().float()
            elif isinstance(out, torch.Tensor):
                self.E = out[:, 1:, :].squeeze(0).detach().cpu().float()

        def _hook_projector(module, inp, out):
            # out: (1, N, dt) or (N, dt)
            if isinstance(out, torch.Tensor):
                self.T = out.view(-1, out.shape[-1]).detach().cpu().float()

        # Locate the vision tower and projector.
        # HuggingFace LlavaForConditionalGeneration layout:
        #   model.vision_tower        — CLIP ViT
        #   model.multi_modal_projector — MLP connector
        llava_model = getattr(self._model, "model", self._model)
        vision_tower = getattr(llava_model, "vision_tower", None)
        projector = getattr(llava_model, "multi_modal_projector", None)

        if vision_tower is not None:
            h = vision_tower.register_forward_hook(_hook_vision_tower)
            self._hooks.append(h)
        else:
            logger.warning("Could not locate vision_tower in model — E will not be captured.")

        if projector is not None:
            h = projector.register_forward_hook(_hook_projector)
            self._hooks.append(h)
        else:
            logger.warning("Could not locate multi_modal_projector — T will not be captured.")

    def _remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# ReconstructionNet / LossEstimator loading
# ---------------------------------------------------------------------------

def load_loss_estimator(
    recon_ckpt: str,
    norm_stats_path: str,
    cfg: dict,
    device: torch.device,
):
    """Load gφ checkpoint + NormStats and return a LossEstimator."""
    # Resolve src path so we can import local modules regardless of cwd.
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from loss_estimator import ReconstructionNet, LossEstimator, NormStats

    loss_cfg = cfg.get("loss", {})
    depth = loss_cfg.get("reconstructor", "mlp3")
    hidden_dim = int(loss_cfg.get("hidden_dim", 2048))
    dropout = float(loss_cfg.get("dropout", 0.1))
    distance = loss_cfg.get("distance", "mse")

    # Infer d_in / d_out from NormStats rather than hard-coding.
    ns = NormStats.load(norm_stats_path)
    d_out = ns.E_mean.shape[0]  # dv — pre-connector dim
    d_in = ns.T_mean.shape[0]   # dt — post-connector dim

    logger.info(
        "ReconstructionNet: depth=%s, d_in=%d, d_out=%d, hidden_dim=%d",
        depth, d_in, d_out, hidden_dim,
    )

    net = ReconstructionNet(
        d_in=d_in,
        d_out=d_out,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    )
    state = torch.load(recon_ckpt, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    net.eval()

    estimator = LossEstimator(net, ns, distance=distance)
    logger.info(
        "LossEstimator ready (distance=%s, norm_stats L_p5=%.4f, L_p95=%.4f).",
        distance, ns.L_p5, ns.L_p95,
    )
    return estimator


# ---------------------------------------------------------------------------
# KNOR computation
# ---------------------------------------------------------------------------

def compute_knor(
    E_pooled_list: List[torch.Tensor],
    T_pooled_list: List[torch.Tensor],
    k: int,
) -> dict:
    """Stack pooled features and call KNORDiagnostic."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from diagnostics.knor import KNORDiagnostic

    E_pooled = torch.stack(E_pooled_list, dim=0)   # (M, dv)
    T_pooled = torch.stack(T_pooled_list, dim=0)   # (M, dt)

    diag = KNORDiagnostic(k=k)
    results = diag.compute(E_pooled, T_pooled)

    # Convert knor_per_image tensor to a plain list for JSON serialisation.
    results["knor_per_image_list"] = results["knor_per_image"].tolist()
    del results["knor_per_image"]   # keep the dict JSON-serialisable

    logger.info(
        "KNOR (k=%d): mean=%.4f, std=%.4f",
        results["k"], results["knor_mean"], results["knor_std"],
    )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_vs_correctness(
    records: List[dict],
    out_path: str,
) -> None:
    """Box/strip plot of mean loss for correct vs incorrect VQA answers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    correct_losses = [r["mean_loss"] for r in records if r["is_correct"]]
    wrong_losses   = [r["mean_loss"] for r in records if not r["is_correct"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [correct_losses, wrong_losses],
        labels=["Correct", "Wrong"],
        patch_artist=True,
        boxprops=dict(facecolor="#a8d8a8", color="#333"),
        medianprops=dict(color="#c0392b", linewidth=2),
    )
    # Overlay individual points (down-sampled if large).
    import random
    rng = random.Random(42)
    for i, group in enumerate([correct_losses, wrong_losses], start=1):
        sample = rng.sample(group, min(200, len(group)))
        ax.scatter(
            [i + rng.uniform(-0.07, 0.07) for _ in sample],
            sample,
            alpha=0.25,
            s=8,
            color="#2c3e50",
            zorder=3,
        )

    ax.set_ylabel("Mean patch reconstruction loss")
    ax.set_title("Loss vs VQA Correctness — VizWiz val (tuning split)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved loss_vs_correctness plot → %s", out_path)


def plot_knor_distribution(
    knor_per_image: List[float],
    knor_mean: float,
    k: int,
    out_path: str,
) -> None:
    """Histogram of per-image KNOR values."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
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
    logger.info("Saved knor_distribution plot → %s", out_path)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def build_summary(
    n_examples: int,
    n_correct: int,
    mean_loss_all: float,
    corr: dict,
    knor: dict,
    elapsed: float,
) -> str:
    """Return a human-readable summary string."""
    vqa_acc = 100.0 * n_correct / max(n_examples, 1)
    lines = [
        "=" * 65,
        "LADP Stage 1 Diagnostic — Summary Report",
        "=" * 65,
        "",
        f"  Examples evaluated   : {n_examples}",
        f"  Correct answers      : {n_correct}  ({vqa_acc:.1f}% VQA accuracy)",
        f"  Mean patch loss (all): {mean_loss_all:.5f}",
        "",
        "--- Loss-Error Correlation ---",
        f"  Pearson  r           : {corr['pearson_r']:+.4f}  (p={corr['pearson_p']:.4g})",
        f"  Spearman r           : {corr['spearman_r']:+.4f}  (p={corr['spearman_p']:.4g})",
        f"  Mean loss (correct)  : {corr['mean_loss_correct']:.5f}",
        f"  Mean loss (wrong)    : {corr['mean_loss_wrong']:.5f}",
        "",
        "--- KNOR Diagnostic ---",
        f"  Neighbourhood k      : {knor['k']}",
        f"  KNOR mean            : {knor['knor_mean']:.4f}",
        f"  KNOR std             : {knor['knor_std']:.4f}",
        "",
    ]

    # Interpretation of Pearson r.
    pr = corr["pearson_r"]
    if pr > 0.3:
        interp = "POSITIVE correlation: higher loss → more errors (supports LADP hypothesis)."
    elif pr < -0.3:
        interp = "NEGATIVE correlation: higher loss → fewer errors (unexpected)."
    else:
        interp = "Weak / no correlation between loss and VQA errors."
    lines += [f"  Interpretation       : {interp}", ""]

    # KNOR interpretation.
    km = knor["knor_mean"]
    if km >= 0.95:
        knor_diag = "Neighbourhood largely preserved (low info-loss risk)."
    elif km >= 0.70:
        knor_diag = "Moderate disruption — connector may lose information."
    else:
        knor_diag = "SEVERE disruption — connector causes significant info loss."
    lines += [f"  KNOR interpretation  : {knor_diag}", ""]

    lines += [
        f"  Elapsed time         : {elapsed:.1f}s",
        "=" * 65,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    device_str = args.device or cfg.get("model", {}).get("device", "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    output_dir = args.output_dir or cfg.get("output_dir", "results/stage1_diagnostic")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    n_examples = args.n_examples or int(cfg.get("data", {}).get("vizwiz_tuning_n", 1000))
    knor_k = args.knor_k

    ann_path = cfg.get("data", {}).get("vizwiz_questions", "data/vizwiz/annotations/val.json")
    img_dir  = cfg.get("data", {}).get("vizwiz_images", "data/vizwiz/images/val")
    precision = cfg.get("model", {}).get("precision", "fp16")
    model_name = cfg.get("model", {}).get("name", "llava-hf/llava-1.5-7b-hf")
    norm_mode = cfg.get("loss", {}).get("norm", "robust")

    # ------------------------------------------------------------------
    # 2. Load annotations
    # ------------------------------------------------------------------
    annotations = load_vizwiz_annotations(ann_path, n_examples)

    # ------------------------------------------------------------------
    # 3. Load gφ + NormStats → LossEstimator
    # ------------------------------------------------------------------
    estimator = load_loss_estimator(
        args.recon_checkpoint,
        args.norm_stats,
        cfg,
        device=device,
    )

    # Import correlate_loss_with_error (uses scipy — must be available).
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from diagnostics.knor import correlate_loss_with_error

    # ------------------------------------------------------------------
    # 4. Load LLaVA-1.5
    # ------------------------------------------------------------------
    llava_model, processor = load_llava(model_name, device, precision=precision)

    # ------------------------------------------------------------------
    # 5. Main inference loop
    # ------------------------------------------------------------------
    from PIL import Image

    records: List[dict] = []
    E_pooled_list: List[torch.Tensor] = []
    T_pooled_list: List[torch.Tensor] = []

    t_start = time.time()

    for idx, ann in enumerate(annotations):
        image_filename: str = ann["image"]
        question: str       = ann["question"]
        gt_answers: List[str] = ann.get("answers", [])

        image_path = os.path.join(img_dir, image_filename)

        # Load image.
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Cannot open image %s: %s — skipping.", image_path, exc)
            continue

        # --- Run LLaVA-1.5 with feature hooks ---
        capture = _FeatureCapture(llava_model)
        with capture:
            pred_answer = llava_answer(llava_model, processor, image, question, device)

        E_raw: Optional[torch.Tensor] = capture.E  # (N, dv) or None
        T_raw: Optional[torch.Tensor] = capture.T  # (N, dt) or None

        # --- Compute per-patch loss ---
        mean_loss: float = float("nan")
        if E_raw is not None and T_raw is not None:
            # Align patch counts (connector may output fewer tokens than encoder)
            n_patches = min(E_raw.shape[0], T_raw.shape[0])
            E_raw = E_raw[:n_patches]
            T_raw = T_raw[:n_patches]

            L = estimator.compute(E_raw, T_raw)   # (N,)
            mean_loss = float(L.mean().item())

            # Collect mean-pooled features for KNOR.
            E_pooled_list.append(E_raw.mean(dim=0).cpu())
            T_pooled_list.append(T_raw.mean(dim=0).cpu())
        else:
            logger.warning(
                "Example %d (%s): feature capture failed (E=%s, T=%s) — loss=NaN.",
                idx, image_filename,
                "ok" if E_raw is not None else "MISSING",
                "ok" if T_raw is not None else "MISSING",
            )

        correct = is_correct(pred_answer, gt_answers)
        acc_score = vizwiz_accuracy(pred_answer, gt_answers)

        records.append(
            {
                "idx": idx,
                "image_id": image_filename,
                "question": question,
                "llava_answer": pred_answer,
                "ground_truth": gt_answers,
                "is_correct": correct,
                "vizwiz_accuracy": acc_score,
                "mean_loss": mean_loss,
            }
        )

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            logger.info(
                "Progress: %d/%d (%.1f it/s) | acc=%.1f%% | mean_loss=%.4f",
                idx + 1, n_examples, rate,
                100.0 * sum(r["is_correct"] for r in records) / len(records),
                sum(r["mean_loss"] for r in records if not (r["mean_loss"] != r["mean_loss"])) / max(1, len(records)),
            )

    total_elapsed = time.time() - t_start
    logger.info("Inference complete in %.1fs for %d examples.", total_elapsed, len(records))

    # ------------------------------------------------------------------
    # 6. Save diagnostic_results.json
    # ------------------------------------------------------------------
    diag_path = os.path.join(output_dir, "diagnostic_results.json")
    with open(diag_path, "w") as fh:
        json.dump(records, fh, indent=2)
    logger.info("Saved diagnostic_results.json → %s", diag_path)

    # ------------------------------------------------------------------
    # 7. KNOR
    # ------------------------------------------------------------------
    knor_results: dict = {}
    if len(E_pooled_list) >= knor_k + 1:
        knor_results = compute_knor(E_pooled_list, T_pooled_list, k=knor_k)
        knor_path = os.path.join(output_dir, "knor_results.json")
        with open(knor_path, "w") as fh:
            json.dump(knor_results, fh, indent=2)
        logger.info("Saved knor_results.json → %s", knor_path)
    else:
        logger.warning(
            "Not enough examples with captured features (%d) to compute KNOR (k=%d). "
            "Need at least %d. Skipping KNOR.",
            len(E_pooled_list), knor_k, knor_k + 1,
        )
        knor_results = {"knor_mean": float("nan"), "knor_std": float("nan"), "k": knor_k}

    # ------------------------------------------------------------------
    # 8. Correlation: loss vs error
    # ------------------------------------------------------------------
    valid_records = [r for r in records if r["mean_loss"] == r["mean_loss"]]  # drop NaN
    losses_list   = [r["mean_loss"] for r in valid_records]
    correct_list  = [r["is_correct"] for r in valid_records]

    corr_results: dict = {}
    if len(valid_records) >= 3:
        corr_results = correlate_loss_with_error(losses_list, correct_list)
        corr_path = os.path.join(output_dir, "correlation_results.json")
        with open(corr_path, "w") as fh:
            json.dump(corr_results, fh, indent=2)
        logger.info("Saved correlation_results.json → %s", corr_path)
    else:
        logger.warning("Too few valid records (%d) for correlation — skipping.", len(valid_records))
        corr_results = {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "mean_loss_correct": float("nan"),
            "mean_loss_wrong": float("nan"),
        }

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    if valid_records:
        plot_loss_vs_correctness(
            valid_records,
            out_path=os.path.join(output_dir, "loss_vs_correctness.png"),
        )

    if "knor_per_image_list" in knor_results:
        plot_knor_distribution(
            knor_results["knor_per_image_list"],
            knor_mean=knor_results["knor_mean"],
            k=knor_results["k"],
            out_path=os.path.join(output_dir, "knor_distribution.png"),
        )

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    n_correct = sum(r["is_correct"] for r in records)
    mean_loss_all = (
        sum(r["mean_loss"] for r in valid_records) / len(valid_records)
        if valid_records else float("nan")
    )

    summary_txt = build_summary(
        n_examples=len(records),
        n_correct=n_correct,
        mean_loss_all=mean_loss_all,
        corr=corr_results,
        knor=knor_results,
        elapsed=total_elapsed,
    )

    summary_path = os.path.join(output_dir, "diagnostic_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(summary_txt)

    # Print to stdout.
    print()
    print(summary_txt)
    logger.info("Saved diagnostic_summary.txt → %s", summary_path)


if __name__ == "__main__":
    main()
