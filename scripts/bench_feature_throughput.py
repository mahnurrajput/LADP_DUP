"""
bench_feature_throughput.py â€” LADP Task 3 measurement
=====================================================
Measures, on this machine, how fast g_phi training data can be produced and
consumed, so the full-COCO option can be chosen with real numbers:

  1. verify   : on-the-fly (E, T) from VisionProjector match the cached
                features (same images, same model) â€” correctness first.
  2. gpu      : CLIP-tower + projector throughput on random pixels (GPU only).
  3. e2e      : end-to-end images/s from JPEG on disk (decode + CLIP
                preprocessing in DataLoader workers + GPU extraction).
  4. gphi     : g_phi training throughput (fwd + bwd + optimiser step) in
                patches/s for mlp3_paper and mlp3_small.

Nothing is trained or written except the JSON report.  Typical runtime: a few
minutes.  Run it while the GPU is otherwise idle; the report records whether
another process was using the GPU.

  python scripts/bench_feature_throughput.py --config configs/base_llava_vizwiz.yaml \
      --hf_home <your Hugging Face cache>      # default: $HF_HOME
  python scripts/bench_feature_throughput.py ... --quick     # ~1 min smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loss_estimator import ReconstructionNet  # noqa: E402
from onthefly_features import ImagePathDataset, VisionProjector  # noqa: E402

N_COCO_TRAIN = 118_287
PATCHES = 576
BYTES_PER_IMAGE_FP16 = PATCHES * (1024 + 4096) * 2  # E + T, fp16


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _other_gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
        me = str(os.getpid())
        return [l.strip() for l in out.splitlines() if l.strip() and not l.startswith(me + ",")]
    except Exception as exc:  # noqa: BLE001
        return [f"unknown ({exc})"]


def verify(vp: VisionProjector, cache_dir: Path, n: int) -> dict:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    rows = [r for r in manifest["files"] if r.get("cached")][:n]
    ds = ImagePathDataset([r["image_path"] for r in rows], vp.image_processor)
    worst_e = worst_t = 0.0
    min_cos = 1.0
    for i, r in enumerate(rows):
        E, T = vp(ds[i].unsqueeze(0))
        ref = torch.load(cache_dir / r["filename"], map_location="cpu", weights_only=False)
        e, t = E[0].float().cpu(), T[0].float().cpu()
        re, rt = ref["E"].float(), ref["T"].float()
        assert e.shape == re.shape and t.shape == rt.shape, (e.shape, re.shape, t.shape, rt.shape)
        worst_e = max(worst_e, (e - re).abs().max().item())
        worst_t = max(worst_t, (t - rt).abs().max().item())
        min_cos = min(min_cos,
                      torch.nn.functional.cosine_similarity(e, re, dim=1).min().item(),
                      torch.nn.functional.cosine_similarity(t, rt, dim=1).min().item())
    # Same images in ONE batch (cache_features.py extracts in batches).
    Eb, Tb = vp(torch.stack([ds[i] for i in range(len(rows))]))
    wb_e = wb_t = 0.0
    for i, r in enumerate(rows):
        ref = torch.load(cache_dir / r["filename"], map_location="cpu", weights_only=False)
        wb_e = max(wb_e, (Eb[i].float().cpu() - ref["E"].float()).abs().max().item())
        wb_t = max(wb_t, (Tb[i].float().cpu() - ref["T"].float()).abs().max().item())
    return {"n_images": len(rows), "max_abs_diff_E": worst_e, "max_abs_diff_T": worst_t,
            "batched_max_abs_diff_E": wb_e, "batched_max_abs_diff_T": wb_t,
            "min_patch_cosine": min_cos,
            "E_abs_scale": float(re.abs().mean()), "T_abs_scale": float(rt.abs().mean())}


def bench_gpu(vp: VisionProjector, batch_sizes, iters: int) -> dict:
    res = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 336, 336, device=vp.device_, dtype=vp.dtype_)
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                vp(x)
            _sync(); t0 = time.perf_counter()
            for _ in range(iters):
                vp(x)
            _sync(); dt = time.perf_counter() - t0
            res[str(bs)] = {"images_per_s": bs * iters / dt,
                            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
        except torch.cuda.OutOfMemoryError:
            res[str(bs)] = {"error": "OOM"}
            torch.cuda.empty_cache()
        del x
    return res


def bench_e2e(vp: VisionProjector, paths, batch_size: int, workers_list) -> dict:
    res = {}
    for w in workers_list:
        ds = ImagePathDataset(paths, vp.image_processor)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=w,
                                         pin_memory=True, persistent_workers=False)
        it = iter(dl)
        vp(next(it)); _sync()                         # warm-up batch (worker start)
        n, t0 = 0, time.perf_counter()
        for px in it:
            vp(px); n += px.shape[0]
        _sync(); dt = time.perf_counter() - t0
        res[str(w)] = {"images_per_s": n / dt, "images_timed": n}
    return res


def bench_gphi(device, batch_patches_list, steps: int) -> dict:
    res = {}
    for depth in ("mlp3_paper", "mlp3_small"):
        net = ReconstructionNet(depth=depth).to(device).train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        for bp in batch_patches_list:
            try:
                t = torch.randn(bp, 4096, device=device)
                e = torch.randn(bp, 1024, device=device)
                for _ in range(2):
                    opt.zero_grad(); torch.nn.functional.mse_loss(net(t), e).backward(); opt.step()
                _sync(); t0 = time.perf_counter()
                for _ in range(steps):
                    opt.zero_grad()
                    loss = torch.nn.functional.mse_loss(net(t), e)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                _sync(); dt = time.perf_counter() - t0
                res[f"{depth}@{bp}"] = {"patches_per_s": bp * steps / dt}
                del t, e
            except torch.cuda.OutOfMemoryError:
                res[f"{depth}@{bp}"] = {"error": "OOM"}
                torch.cuda.empty_cache()
        del net, opt
        torch.cuda.empty_cache()
    return res


def _paper_step_fn(device):
    net = ReconstructionNet(depth="mlp3_paper").to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)

    def step(T: torch.Tensor, E: torch.Tensor) -> None:
        # standardisation with fixed dummy stats (cost-equivalent to the real one)
        T = (T.float() - 0.0) / 1.0
        E = (E.float() - 0.0) / 1.0
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(T), E)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return step


def bench_cache_train_loop(cache_dir: Path, n_images: int, start: int, images_per_step: int) -> dict:
    """(a) real loop: ShardedPatchLoader over cached files â†’ mlp3_paper steps.
    Files start at index ``start`` (pick ones not read recently: the page cache
    can still serve some of them â€” compare with bench_disk_read.py)."""
    from loss_estimator import ShardedPatchLoader
    files = sorted(str(p) for p in cache_dir.glob("features_*.pt"))[start:start + n_images]
    dev = torch.device("cuda")
    step = _paper_step_fn(dev)
    loader = ShardedPatchLoader(files, images_per_step * PATCHES, images_per_step,
                                shuffle=True, seed=0, device=dev)
    it = iter(loader)
    e, t = next(it); step(t, e); _sync()                 # warm-up (first shard)
    n, t0 = images_per_step, time.perf_counter()
    for e, t in it:
        step(t, e); n += e.shape[0] // PATCHES
    _sync(); dt = time.perf_counter() - t0
    timed = n - images_per_step
    return {"images_timed": timed, "seconds": dt, "images_per_s": timed / dt,
            "patches_per_s": timed * PATCHES / dt, "first_index": start}


def bench_onthefly_train_loop(vp_args: dict, img_paths: list, images_per_step: int, workers: int) -> dict:
    """(b) real loop: JPEG â†’ workers â†’ VisionProjector â†’ mlp3_paper step."""
    dev = torch.device("cuda")
    vp = VisionProjector(**vp_args)
    step = _paper_step_fn(dev)
    ds = ImagePathDataset(img_paths, vp.image_processor)
    dl = torch.utils.data.DataLoader(ds, batch_size=images_per_step, num_workers=workers, pin_memory=True)
    it = iter(dl)

    def one(px):
        E, T = vp(px)
        step(T.reshape(-1, T.shape[-1]), E.reshape(-1, E.shape[-1]))

    one(next(it)); _sync()
    n, t0 = 0, time.perf_counter()
    for px in it:
        one(px); n += px.shape[0]
    _sync(); dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    del vp
    torch.cuda.empty_cache()
    return {"images_timed": n, "seconds": dt, "images_per_s": n / dt,
            "patches_per_s": n * PATCHES / dt, "workers": workers, "peak_mem_gib": peak}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--n_images", type=int, default=1536, help="images for the end-to-end test")
    ap.add_argument("--batch_sizes", default="1,16,32,64,128")
    ap.add_argument("--e2e_batch_sizes", default="1,32",
                    help="end-to-end batch sizes (1 = what cache_features.py uses by default)")
    ap.add_argument("--workers", default="4,8,12")
    ap.add_argument("--verify", type=int, default=32)
    ap.add_argument("--out", default="results/throughput/bench_feature_throughput.json")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (numbers not meaningful)")
    ap.add_argument("--loop_images", type=int, default=1536,
                    help="images per training-loop benchmark (12 steps of 128)")
    ap.add_argument("--loop_cache_start", type=int, default=0,
                    help="first cache file for loop (a)")
    ap.add_argument("--loop_jpeg_start", type=int, default=20000,
                    help="first COCO image for loop (b) (outside the cached 10k, not read recently)")
    ap.add_argument("--loop_workers", type=int, default=12)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    proj = Path(args.config).resolve().parent.parent
    cache_dir = proj / cfg["data"]["cache_dir"] / "coco_train"
    img_dir = proj / cfg["data"]["coco_train_images"]
    if args.quick:
        args.n_images, args.batch_sizes, args.workers, args.verify = 96, "16", "4", 2
    bss = [int(x) for x in args.batch_sizes.split(",")]
    wl = [int(x) for x in args.workers.split(",")]

    report = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": args.quick,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "other_gpu_processes_at_start": _other_gpu_processes()}
    print("Other GPU processes:", report["other_gpu_processes_at_start"] or "none")

    t0 = time.perf_counter()
    vp = VisionProjector(cfg["model"]["name"], hf_home=args.hf_home)
    report["load_s"] = time.perf_counter() - t0
    report["vision_projector_params"] = sum(p.numel() for p in vp.parameters())
    print(f"Loaded vision tower + projector in {report['load_s']:.1f}s "
          f"({report['vision_projector_params']:,} params)")

    report["verify"] = verify(vp, cache_dir, args.verify)
    print("verify:", report["verify"])

    report["gpu_only"] = bench_gpu(vp, bss, iters=3 if args.quick else 20)
    print("gpu_only:", report["gpu_only"])
    best_bs = max((b for b in bss if "images_per_s" in report["gpu_only"][str(b)]),
                  key=lambda b: report["gpu_only"][str(b)]["images_per_s"])

    all_jpg = sorted(str(p) for p in img_dir.glob("*.jpg"))
    report["e2e"] = {}
    e2e_bss = [1] if args.quick else [int(x) for x in args.e2e_batch_sizes.split(",")]
    for k, ebs in enumerate(e2e_bss):
        # different images per setting, outside the cached 10k (cold-ish reads)
        off = 30000 + k * (args.n_images + 128) * len(wl)
        res = {}
        for j, w in enumerate(wl):
            o = off + j * (args.n_images + ebs)
            res.update(bench_e2e(vp, all_jpg[o:o + args.n_images + ebs], ebs, [w]))
        report["e2e"][f"bs{ebs}"] = res
    print("e2e:", report["e2e"])
    del vp; torch.cuda.empty_cache()

    gbp = [128, 8192] if args.quick else [128, 4096, 36864, 73728]
    report["gphi_train"] = bench_gphi(torch.device("cuda"), gbp, steps=3 if args.quick else 10)
    print("gphi_train:", report["gphi_train"])

    # ---- (a) / (b) real training loops, batch = 128 images, mlp3_paper ---------------
    ips = 16 if args.quick else 128
    n_loop = 3 * ips if args.quick else args.loop_images
    report["loop_cache"] = bench_cache_train_loop(cache_dir, n_loop, args.loop_cache_start, ips)
    print("loop (a) cache-fed:", report["loop_cache"])
    jpegs = sorted(str(p) for p in img_dir.glob("*.jpg"))[args.loop_jpeg_start:args.loop_jpeg_start + n_loop + ips]
    report["loop_onthefly"] = bench_onthefly_train_loop(
        {"model_name": cfg["model"]["name"], "hf_home": args.hf_home}, jpegs, ips,
        4 if args.quick else args.loop_workers)
    print("loop (b) on-the-fly:", report["loop_onthefly"])

    # ---- estimates (per epoch, full COCO train2017) ---------------------------------
    e2e_rate = max(v["images_per_s"] for v in report["e2e"]["bs1"].values())   # cache build (bs 1)
    n_val = int(N_COCO_TRAIN * 0.05)
    n_train = N_COCO_TRAIN - n_val
    est = {"n_images": N_COCO_TRAIN, "n_train_images": n_train, "n_val_images": n_val,
           "steps_per_epoch_at_128_images": math.ceil(n_train / 128),
           "extract_once_s": N_COCO_TRAIN / e2e_rate,
           "full_cache_GB": N_COCO_TRAIN * BYTES_PER_IMAGE_FP16 / 1e9}
    for k, v in report["gphi_train"].items():
        if "patches_per_s" in v:
            est[f"gphi_compute_per_epoch_s[{k}]"] = n_train * PATCHES / v["patches_per_s"]
    est["onthefly_extract_per_epoch_s"] = N_COCO_TRAIN / e2e_rate   # train + val images
    # whole epoch = train images through the loop + val images (forward only, ~1/3 of a step;
    # bounded above by the loop rate)
    for tag in ("loop_cache", "loop_onthefly"):
        r = report[tag]["images_per_s"]
        est[f"epoch_upper_bound_min[{tag}]"] = (n_train + n_val) / r / 60
    report["estimates"] = est
    report["other_gpu_processes_at_end"] = _other_gpu_processes()
    print("estimates:", json.dumps(est, indent=1))

    out = proj / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("Report written to", out)


if __name__ == "__main__":
    main()
