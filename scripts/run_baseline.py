"""Baseline LLaVA-1.5-7B (no LADP) on VizWiz VQA val.

Writes one JSON line per example (resumable) to results/baseline/predictions.jsonl
and, at the end, a summary.json.  Split: first `--heldout-start` examples = tuning,
the rest = held-out.  The summary prints TUNING numbers only; held-out numbers are
stored per-example but must not be inspected until hyper-parameters are frozen.

Usage: python scripts/run_baseline.py [--limit N] [--heldout-start 3800]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "E:/hf_cache")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from src.connector_adapter import LlavaAdapter

PROMPT = ("USER: <image>\n{q} When the provided information is insufficient, "
          "respond with 'Unanswerable'. Answer the question using a single word or phrase. ASSISTANT:")

# --- VQA answer normalisation (standard VQA-eval style) ---------------------
_NUM = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ART = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$:|~]")


def norm(a: str) -> str:
    a = a.replace("\n", " ").replace("\t", " ").strip().lower()
    a = _PUNCT.sub(" ", a)
    a = re.sub(r"(?<!\d)\.(?!\d)", "", a)  # keep decimal points
    words = [_NUM.get(w, w) for w in a.split() if w not in _ART]
    return " ".join(words)


def vqa_acc(pred: str, answers: list[str]) -> float:
    """Official VQA accuracy: mean over the 10 leave-one-out subsets of min(1, #match/3)."""
    p = norm(pred)
    gts = [norm(a) for a in answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return sum(accs) / len(accs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--heldout-start", type=int, default=3800)
    ap.add_argument("--out", default=str(ROOT / "results/baseline"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    data = json.load(open(ROOT / "data/vizwiz/val.json", encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]
    img_dir = ROOT / "data/vizwiz/val"

    done = {}
    if pred_path.exists():
        for line in open(pred_path, encoding="utf-8"):
            r = json.loads(line); done[r["idx"]] = r
    print(f"{len(data)} examples, {len(done)} already done", flush=True)

    ad = LlavaAdapter(device="cuda", precision="fp16")
    model, proc = ad.get_model(), ad.get_processor()
    torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    with open(pred_path, "a", encoding="utf-8") as fh:
        for idx, ex in enumerate(data):
            if idx in done:
                continue
            img = Image.open(img_dir / ex["image"]).convert("RGB")
            inp = proc(images=img, text=PROMPT.format(q=ex["question"]), return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
            torch.cuda.synchronize(); t = time.time()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=16, do_sample=False)
            torch.cuda.synchronize(); lat = time.time() - t
            pred = proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rec = {"idx": idx, "image": ex["image"], "question": ex["question"], "pred": pred,
                   "answer_type": ex.get("answer_type"), "answerable": ex.get("answerable"),
                   "acc": vqa_acc(pred, [a["answer"] for a in ex["answers"]]),
                   "latency_ms": lat * 1000, "split": "heldout" if idx >= args.heldout_start else "tuning"}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            done[idx] = rec
            if len(done) % 100 == 0:
                tun = [r["acc"] for r in done.values() if r["split"] == "tuning"]
                el = time.time() - t_start
                print(f"[{len(done)}/{len(data)}] tuning acc so far={sum(tun)/max(1,len(tun)):.3f}  "
                      f"elapsed={el/60:.1f} min", flush=True)

    # ---- summary: TUNING only ------------------------------------------------
    tun = [r for r in done.values() if r["split"] == "tuning"]
    lats = sorted(r["latency_ms"] for r in done.values())
    by_type: dict[str, list[float]] = {}
    for r in tun:
        by_type.setdefault(r["answer_type"], []).append(r["acc"])
    summary = {
        "n_total": len(done), "n_tuning": len(tun), "n_heldout": len(done) - len(tun),
        "tuning_vqa_acc": sum(r["acc"] for r in tun) / len(tun),
        "tuning_by_answer_type": {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in by_type.items()},
        "latency_ms_median": lats[len(lats) // 2], "latency_ms_p95": lats[int(len(lats) * 0.95)],
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "visual_tokens": 576, "prompt": PROMPT, "heldout_start": args.heldout_start,
        "note": "held-out accuracy intentionally not reported here",
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
