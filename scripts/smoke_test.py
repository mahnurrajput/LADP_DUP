"""Stage-0/1 smoke test: model loads, E/T extraction is correct, baseline VQA runs.

Usage:  python scripts/smoke_test.py [--n 10]
"""
from __future__ import annotations

import argparse
import json
import os
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


def vqa_score(pred: str, answers: list[str]) -> float:
    pred = pred.strip().lower().rstrip(".")
    ans = [a.strip().lower() for a in answers]
    return min(1.0, sum(a == pred for a in ans) / 3.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"gpu={torch.cuda.get_device_name(0)}")
    import transformers
    print("transformers", transformers.__version__)

    t0 = time.time()
    ad = LlavaAdapter(device="cuda", precision="fp16")
    print(f"[1] model loaded in {time.time()-t0:.0f}s  "
          f"VRAM={torch.cuda.memory_allocated()/2**30:.1f} GiB")

    ad.smoke_test()
    print("[2] adapter.smoke_test (synthetic image) PASSED")

    val = json.load(open(ROOT / "data/vizwiz/val.json", encoding="utf-8"))[: args.n]
    img_dir = ROOT / "data/vizwiz/val"

    # [3] correctness of the E/T hooks against the model's own image path
    img0 = Image.open(img_dir / val[0]["image"]).convert("RGB")
    f = ad.extract_features(img0)
    model, proc = ad.get_model(), ad.get_processor()
    pv = proc.image_processor(img0, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
    with torch.no_grad():
        ref = model.get_image_features(
            pixel_values=pv,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        )
    ref = ref[0] if isinstance(ref, (list, tuple)) else ref
    ref = ref.reshape(-1, ref.shape[-1])
    diff = (ref.float() - f["T"].float()).abs().max().item()
    print(f"[3] E={tuple(f['E'].shape)} T={tuple(f['T'].shape)}  "
          f"max|T - model.get_image_features| = {diff:.2e}  "
          f"{'PASSED' if diff < 1e-2 else 'MISMATCH'}")

    # [4] baseline answers
    scores, lat = [], []
    for ex in val:
        img = Image.open(img_dir / ex["image"]).convert("RGB")
        inp = proc(images=img, text=PROMPT.format(q=ex["question"]), return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize(); lat.append(time.time() - t)
        pred = proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        s = vqa_score(pred, [a["answer"] for a in ex["answers"]])
        scores.append(s)
        print(f"   {ex['image']}  Q={ex['question'][:50]!r}  pred={pred!r}  gt0={ex['answers'][0]['answer']!r}  score={s:.2f}")
    print(f"[4] baseline on {len(val)} examples: mean VQA={sum(scores)/len(scores):.3f}  "
          f"median latency={sorted(lat)[len(lat)//2]*1000:.0f} ms  "
          f"peak VRAM={torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    print("SMOKE TEST DONE")


if __name__ == "__main__":
    main()
