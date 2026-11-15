# LADP — Loss-Aware Dynamic Projection for Vision-Language Models

Final Year Project **F26-188**, FAST NUCES Lahore.
Fatima Riaz · Mahnoor Naveed · Faryal Zahra — supervised by Dr. Maryam Bashir.

LADP studies **where information is lost in the vision → language connector** of a
frozen vision-language model and uses that signal to decide which visual tokens to
preserve. For every image patch it estimates

- **connector loss** `L_i` — how badly the patch's pre-connector CLIP feature `E_i`
  can be reconstructed from its post-connector token `T_i`, and
- **question relevance** `R_i` — how relevant the patch is to the question,

and combines them into a priority score `S_i = α·L̄_i + β·R̄_i` (or `L̄_i·R̄_i`) used
to preserve high-priority tokens. Experiments use **LLaVA-1.5-7B** (frozen) on
**VizWiz**, whose images are taken by blind and low-vision users.

This is an offline research prototype. Only public datasets and models are used.

## Pipeline

```
Image ─► CLIP ViT-L/14-336 ─► E (576×1024) ─► projector P_θ ─► T (576×4096) ─► LLM
                                  │                                │
                                  │          g_φ(T_i) = Ê_i ◄──────┘
                                  └──► L_i = ‖E_i − Ê_i‖²  (robust p5/p95 norm)
Question ─► CLIP text encoder ─► R_i = cos(E_i, q)
                     S_i = α·L̄_i + β·R̄_i  ─►  preservation strategy (HT | SR | HRR)
```

`E` is taken from the penultimate vision layer (`vision_feature_layer = -2`, CLS
removed). `g_φ` is the **only trained component**; LLaVA and CLIP stay frozen.

## Reconstruction architectures and branches

Two architectures for g_φ are kept.

| | `mlp3_small` (old, pilot) | `mlp3_paper` (new, **primary**) |
|---|---|---|
| Parameters | **14,685,184** | **27,282,432** |
| Layers | 4096 → 2048 → 2048 → 1024, GELU + dropout 0.1 | 4096 → 4096 → 2048 → 1024, LayerNorm + GELU + dropout 0.1 after each hidden layer |
| Initialisation | Kaiming uniform, zero bias | PyTorch default (as in the source code) |
| Training setup | Adam, constant lr 1e-4, summed MSE, no clipping, patience 5 | AdamW, 2000-step warm-up + cosine, mean MSE, clip 1.0, patience 10 / min_delta 1e-3 |
| Purpose | FYP-1 pilot (10k COCO images); used for the FYP-1 Stage-1 diagnostic; ablation | Same model as `EmbeddingTransformer(num_layers=3)` in the Lost in Embeddings code; target for full COCO |
| Config | `configs/pilot_mlp3_small.yaml` | `configs/base_llava_vizwiz.yaml` (full COCO), `configs/paper_10k_mlp3_paper.yaml` (pilot's 10k images) |

`mlp3_paper` is the primary, paper-based implementation. `mlp3_small` is kept so the
pilot can be reproduced and compared.

### Branches

| Branch | Contains |
|---|---|
| `main` | The original implementation: the small model only (named `mlp3`, 14,685,184 params), before resume support. Kept unchanged. |
| `feat/faryal/mlp-small-reconstruction-module` | `main` + exactly the code that trained the FYP-1 pilot (commit `9a488d8`: `mlp3` with resume support), the pilot's validation split and `docs/PILOT_MLP3_SMALL.md`. |
| `feat/faryal/mlp-27.3-reconstruction-module` (this branch) | Built on the small-model branch. Adds `mlp3_paper` (27,282,432 params) next to `mlp3_small` (alias `mlp3`), the paper training setup (AdamW, warm-up + cosine, clipping), the audit fixes F1–F11 (`docs/AUDIT_mlp3_paper.md`, `docs/CHANGES_audit_fixes.md`), the resume fixes R1/R3 (`docs/CHANGES_R1_R3.md`), the resumable / failure-aware feature cache, the configs above and the tests. |

### Training each architecture (on this branch)

```bash
# mlp3_paper on full COCO (needs the complete 118,287-image cache; refuses anything less)
python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml
# mlp3_paper on the pilot's 10k images (same split as the pilot)
python scripts/train_recon.py --config configs/paper_10k_mlp3_paper.yaml
# mlp3_small: reproduce the pilot
python scripts/train_recon.py --config configs/pilot_mlp3_small.yaml
```

Checkpoints go to `checkpoints/<training.run_name>/`: `recon_full_mlp3_paper/`,
`recon_paper_10k_mlp3_paper/` and `recon_pilot/`.

> **Warning:** the pilot config writes to `checkpoints/recon_pilot/` and
> `results/pilot_mlp3_small/`, where the finished pilot is stored. Back these up, or
> change `training.run_name` and `output_dir`, before retraining with it. On `main`
> and on the small-model branch,
> `python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml` trains
> `mlp3` into `checkpoints/recon/` and `results/stage1_diagnostic/`, which also hold
> the original pilot run.

### Checkpoints are not interchangeable

A `mlp3_small` checkpoint (`checkpoints/recon_pilot/best.pt`: `net.0` 2048×4096,
`net.3` 2048×2048, `net.6` 1024×2048) cannot be loaded into `mlp3_paper`
(`net.0` 4096×4096, LayerNorms at `net.1` / `net.5`), and the reverse fails too:
`load_state_dict` raises a size/key error. Each checkpoint must be used with the
config of its own architecture. The `norm_stats.pt` (mean/std, p5/p95) also belongs
to the model it was trained with. The pilot checkpoint loads into `mlp3_small` on
both branches.

## Repository layout

| Path | Contents |
|---|---|
| `configs/base_llava_vizwiz.yaml` | Target run: `mlp3_paper` as in Lost in Embeddings, **full COCO 2017 train (118,287 images, all patches)**, batch 128 images |
| `configs/paper_10k_mlp3_paper.yaml` | Same paper setup on exactly the pilot's 10k images and split |
| `configs/splits/coco_pilot_val500.json` | The pilot's 500 validation images, pinned in every run (`training.fixed_val_images`) |
| `configs/pilot_mlp3_small.yaml` | The FYP-1 pilot g_φ (`mlp3_small`, Adam), used for the FYP-1 Stage-1 diagnostic |
| `src/connector_adapter.py` | `LlavaAdapter` — extracts `E` and `T` from LLaVA-1.5 |
| `src/onthefly_features.py` | `VisionProjector` — CLIP tower + projector only (no LLM), for on-the-fly `E`/`T` |
| `src/loss_estimator.py` | `ReconstructionNet` (g_φ), `LossEstimator`, `NormStats`, `ShardedPatchLoader`, `ReconstructionTrainer` |
| `src/relevance_scorer.py` | `CLIPRelevanceScorer` (default) and `LLMEmbeddingRelevanceScorer` (ablation) |
| `src/normaliser.py`, `src/priority_module.py` | Score normalisation and priority combination |
| `src/diagnostics/knor.py` | KNOR neighbourhood diagnostic and loss–error correlation |
| `scripts/` | Command-line entry points (below) |
| `tests/` | Unit tests (pytest) |
| `results/` | Run outputs (baseline, Stage-1 diagnostic) |

## Setup

Tested on Windows 11, Python 3.10, CUDA 12.8, NVIDIA RTX A4000 (16 GB).
LLaVA-1.5-7B in fp16 needs about 14 GB of GPU memory.

```bash
conda create -n ladp python=3.10
conda activate ladp
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Store the Hugging Face model cache on a drive with space (~14 GB):

```bash
set HF_HOME=E:\hf_cache        # Windows cmd;  PowerShell: $env:HF_HOME="E:\hf_cache"
```

### Data (not included in the repository)

Download separately and place under `data/`:

```
data/
  coco/train2017/          COCO 2017 train images (used to train g_φ)
  vizwiz/val/              VizWiz VQA val images
  vizwiz/val.json          VizWiz VQA val annotations
  vizwiz_grounding/        VizWiz answer-grounding masks (relevance sanity check)
  cache/                   generated by cache_features.py
```

## Running the pipeline

All commands are run from the repository root.

```bash
# 0. Check the environment and model
python scripts/smoke_test.py --n 10

# 1. Baseline LLaVA-1.5 accuracy on VizWiz val
python scripts/run_baseline.py

# 2. Cache E/T features (5,900,261 bytes per image; full COCO = 697.92 GB).
#    Resumable and idempotent: reuses and verifies existing files, writes atomically,
#    stops cleanly below 40 GB free (--min_free_gb). Image count = data.max_images.
python scripts/cache_features.py --config configs/base_llava_vizwiz.yaml --split coco_train

# 3. Train the reconstruction network g_φ
python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml
#    resume after an interruption (full state, exact):
python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml --resume checkpoints/recon_full_mlp3_paper/last.pt

# 4. Stage-1 diagnostic: does connector loss predict answer errors?
#    NOTE: run_diagnostic.py on this branch is the ORIGINAL version. Its known bugs
#    (E taken from the wrong CLIP layer, crash on VizWiz answer dicts, prompt and
#    metric different from the baseline) are fixed in separate, not-yet-merged work.
#    Do not use it for results from this branch.
python scripts/run_diagnostic.py --config configs/pilot_mlp3_small.yaml

# Throughput of on-the-fly feature extraction and g_φ training (a few minutes)
python scripts/bench_feature_throughput.py --config configs/base_llava_vizwiz.yaml   # uses $HF_HOME
# Unbuffered sequential read speed of the cache drive
python scripts/bench_disk_read.py --gb 20
```

| Script | Writes |
|---|---|
| `cache_features.py` | `data/cache/<split>/features_XXXXXX.pt` (`E`, `T` in fp16), `manifest.json` |
| `train_recon.py` | `<checkpoints_dir>/<training.run_name>/{best.pt, last.pt, norm_stats.pt}`, `<output_dir>/{train_recon.log, recon_loss_curve.png, config_snapshot.yaml}` |
| `run_diagnostic.py` | `<output_dir>/{diagnostic_results.json, knor_results.json, correlation_results.json, loss_vs_correctness.png, knor_distribution.png, diagnostic_summary.txt}` |
| `bench_feature_throughput.py` | `results/throughput/bench_feature_throughput.json` |

### g_φ variants

| `loss.reconstructor` | Layers (LLaVA) | Params | Use |
|---|---|---|---|
| `mlp3_small` (alias `mlp3`) | 4096 → 2048 → 2048 → 1024, GELU, dropout 0.1 | 14,685,184 | FYP-1 pilot; ablation (`configs/pilot_mlp3_small.yaml`) |
| `mlp3_paper` | 4096 → 4096 → 2048 → 1024, LayerNorm + GELU + dropout 0.1 after each hidden layer | 27,282,432 | primary; identical to `EmbeddingTransformer(num_layers=3)` in the Lost in Embeddings code (`configs/base_llava_vizwiz.yaml`) |
| `mlp1`, `mlp2` | linear / one hidden layer | — | ablation |

`loss.reconstructor` must be set explicitly (there is no default), and
`loss.expected_params` pins the size: the run stops if the built model differs.

### Training options (`training.*`)

| Key | Pilot (`pilot_mlp3_small.yaml`) | Paper setup (`base_llava_vizwiz.yaml`) | Source |
|---|---|---|---|
| `optimizer` | `adam` | `adamw` (weight decay = torch default 0.01) | `optim.AdamW(lr)` |
| `scheduler` | `none` | `warmup_cosine`: linear warm-up `warmup_steps` 2000, cosine to `min_lr_ratio` 0.01 | `SequentialLR(LambdaLR, CosineAnnealingLR)` |
| `grad_clip` | `null` | `1.0` | `clip_grad_norm_(max_norm=1.0)` |
| `loss_reduction` | `sum` | `mean` | `nn.MSELoss()` |
| `patience` / `min_delta` | 5 / 0 | 10 / 1e-3 | 10 / 1e-3 |
| `lr`, `epochs`, `dropout` | 1e-4, 30, 0.1 | 1e-4, 30, 0.1 | paper §4.3 |
| `batch_size` | 128 patches | 73,728 patches = 128 images (`files_per_shard: 128`) | 128 **images** per micro-batch, ×4 gradient accumulation (not replicated) |
| `warmup_steps` / `warmup_ratio` | — | 2000 (7.6% of 26,340 steps); small runs: `warmup_ratio` 0.05 | 2000 steps |
| `fixed_val_images` | — | pilot's 500 val images pinned; also logged as `val_ref_loss` | — |

Other keys: `run_name` (checkpoint sub-folder), `data.max_images` (COCO images used,
train + val; `null` = all cached), `val_fraction`, `files_per_shard`, `seed`.

Safety checks (added after the implementation audit, `docs/AUDIT_mlp3_paper.md`):

| Key / behaviour | Effect |
|---|---|
| `loss.reconstructor` | **Required.** A missing or misspelt key stops the run (it used to fall back to `mlp3_small`). |
| `loss.expected_params` | The built model must have exactly this many parameters (27,282,432 for `mlp3_paper`). |
| `data.expected_images` | With `max_images: null`, the cache must hold exactly this many images (118,287). Required together with `require_complete_cache`. |
| `data.require_complete_cache` | The manifest must say `"complete": true`; older manifests without that flag (the 10k pilot cache) are refused. |
| `training.allow_resume_mismatch` | Default `false`: resume refuses a `last.pt` made with a different split, batch, model or optimiser setup. |
| steps/epoch | Counted per shard (batches never cross shards), so the lr schedule ends on the last step; a warning is logged if `files_per_shard × 576` is not a multiple of `batch_size`. |
| resume | Reuses saved normalisation statistics (`train_stats.pt`, also in `last.pt`) when the train files are unchanged; a weights-only resume continues the lr schedule instead of replaying warm-up; a checkpoint whose run already finished or early-stopped only finalises (p5/p95, `norm_stats.pt`). |
| epoch extension (R1) | Resuming `last.pt` with more `epochs` is allowed (also with `warmup_ratio`): warm-up stays at the steps already used, and the cosine continues from the current lr down to the floor at the new end, with no jump. Changing the configured warm-up is refused. |
| `best.pt` integrity (R3) | `last.pt` records a fingerprint of the weights in `best.pt`. A weights-only resume makes the resumed weights `best.pt` (a different existing file is kept as `best.superseded_<time>.pt`); a full resume refuses a foreign `best.pt`; the final reload is verified before `norm_stats.pt` is fitted. |
| cache completeness | `cache_features.py` records every image as cached / failed (with attempts) / excluded (explicit, `--exclude_failed`) / pending; `complete` only when all are cached or explicitly excluded. |
| exit codes | `train_recon.py` exits 2 on a refused config/cache/resume, 3 if the post-training sanity check fails. |

### Training notes

- g_φ is trained with MSE in standardised feature space (dataset mean/std of the
  training split), as in the source paper.
- The train/validation split is **by image** (5 % of images held out), so patches
  of one image never appear in both splits. The source code also splits by image
  (one dataset item = one image's 576 patches) but uses 80/20.
- Resuming from `last.pt` restores model, optimiser, scheduler, epoch, best loss,
  early-stopping counter, loss history and RNG state, so it reproduces an
  uninterrupted run exactly.
- Features are streamed in shards of whole files (`training.files_per_shard`,
  default 64) with patches shuffled across the images in each shard; normalisation
  statistics are computed in a single streaming pass, so RAM use does not grow
  with the dataset size.
- The cache is read in full every epoch (~59 GB for 10k images), so epoch time is
  usually limited by disk read speed.

## Tests

```bash
python -m pytest tests -q          # 237 tests on this branch (real-data tests skip if the cache, weights or GPU are missing)
```

## Status

| Stage | Status |
|---|---|
| Baseline on VizWiz val (4,319 examples; tuning VQA accuracy 0.528) | Done |
| COCO feature cache (10,000 images, validated) | Done |
| g_φ pilot (`mlp3_small`, Adam) on 10k images | Done (best val MSE 0.072404, epoch 16) |
| g_φ `mlp3_paper` + AdamW/warm-up/cosine | Implemented, audited and unit-tested; not trained yet |
| Full-COCO cache (118,287 images) | Planned: needs ≥ 61.3 GB more free space on E: |
| On-the-fly E/T extraction (`VisionProjector`) | Implemented; matches cached features exactly |
| Stage-1 diagnostic (KNOR, loss–error correlation) | Pending |
| Preservation strategies (HT, SR, HRR) | Not started |

## Datasets and models

- [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) and
  [OpenAI CLIP ViT-L/14-336](https://github.com/openai/CLIP), via Hugging Face
  Transformers and OpenCLIP.
- [COCO 2017](https://cocodataset.org/), [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/)
  and [VizWiz answer grounding](https://vizwiz.org/tasks-and-datasets/answer-grounding/).

Datasets are used under their own licences and are not redistributed here.
