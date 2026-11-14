# Implementation audit: `mlp3_paper` (27.3M) and the cached-feature training path

Project F26-188 (LADP). Audit date: 2026-11-14. Audit only: no code was edited, no
training was run, no cache was built, and nothing was committed.

**Reference:** Lost in Embeddings code, `lyan62/vlm-info-loss`,
`scripts/reconstruct_embeddings_parallel.py` (`EmbeddingTransformer`, `num_layers=3`),
re-read on 2026-11-14.

**How it was checked:** the code path was followed from the YAML file →
`scripts/train_recon.py:340-351` → `ReconstructionNet.__init__`
(`src/loss_estimator.py:139-157`) → `ReconstructionTrainer` → `best.pt`, and each
config was loaded through the real constructors.

## Summary

| Question | Answer |
|---|---|
| A. Is the 27.3M architecture fully implemented? | **YES** |
| B. Parameter count, actual vs expected | **27,282,432 / 27,282,432** |
| D. Does the active config use `mlp3_paper`? | **YES** (`configs/base_llava_vizwiz.yaml`) |
| E. Is the pipeline ready for full cached-feature training? | **NO**: not enough disk space, and the completeness check passes on the 10k cache (F1) |

## A. Architecture

`src/loss_estimator.py:139-151` builds:

```
Linear(4096→4096) → LayerNorm(4096) → GELU → Dropout(0.1)
Linear(4096→2048) → LayerNorm(2048) → GELU → Dropout(0.1)
Linear(2048→1024)
```

- **Hidden width:** `h = min(hidden_dim, max(d_in, d_out))`, the same as the source. It gives 2048 for `hidden_dim: 2048`.
- **Initialisation:** PyTorch default `nn.Linear` init (`_init_weights` is skipped for `mlp3_paper`, line 156). The source class has no custom init either.
- **Residual connections:** none, in both the source and our code.

## B. Parameter count

| | Count |
|---|---|
| By hand: (4096·4096+4096) + 2·4096 + (4096·2048+2048) + 2·2048 + (2048·1024+1024) | 27,282,432 |
| Model built from `base_llava_vizwiz.yaml` | 27,282,432 |
| Model built from `paper_10k_mlp3_paper.yaml` | 27,282,432 |
| Pilot `mlp3_small` (`pilot_mlp3_small.yaml`) | 14,685,184 |

The pilot's `best.pt` fails to load into `mlp3_paper` (RuntimeError from the strict
load), so the two models cannot be mixed up silently.

## C. Comparison with the reference

### Matches (checked in code)

| Component | Ours | Source |
|---|---|---|
| Layers, sizes, LayerNorm, GELU, dropout 0.1 | yes | yes |
| Initialisation | PyTorch default | PyTorch default |
| Optimiser | AdamW, weight decay 0.01 (torch default), all parameters | `optim.AdamW(model.parameters(), lr)` |
| Warm-up | linear, 2000 steps | `LambdaLR(min(1, s/2000))` |
| Decay | cosine to 0.01·lr, one scheduler step per optimiser step | `CosineAnnealingLR(eta_min=0.01·lr)` |
| Gradient clipping | `clip_grad_norm_(1.0)` | `clip_grad_norm_(max_norm=1.0)` |
| Loss | `F.mse_loss(reduction="mean")` | `nn.MSELoss()` |
| Early stopping | `val < best − 1e-3`, patience 10 | same |
| Split | by image | by image (one dataset item = one image) |

### Does not match

| Item | Ours | Source |
|---|---|---|
| Mixed precision | fp32 (`loss_estimator.py:1086-1102`) | AMP (`autocast` + `GradScaler`) |
| Gradient accumulation | none (documented in the config) | 4 batches per optimiser step |
| Learning-rate curve | cosine decays fully over the run | schedule length counts batches but steps per optimiser step, so the lr decays only about ¼ of the way (from reading the code; not run) |
| Validation split | 5 % of images | 80/20 |
| Training settings | lr 1e-4, 30 epochs, 3 layers, 128 images per batch (paper text) | script defaults: lr 3e-4, 20 epochs, `num_layers` 16, batch 64. **NOT CONFIRMED:** the command line the authors actually ran |

## D. Configuration as loaded

| Config | depth | params | Optimiser / schedule | Clip | Loss | Patience / min_delta | Batch (patches) | max_images |
|---|---|---|---|---|---|---|---|---|
| `base_llava_vizwiz.yaml` | mlp3_paper | 27,282,432 | AdamW 0.01, warmup_cosine 2000 | 1.0 | mean | 10 / 1e-3 | 73,728 | null (all) |
| `paper_10k_mlp3_paper.yaml` | mlp3_paper | 27,282,432 | AdamW 0.01, warmup_cosine, warm-up ratio 0.05 | 1.0 | mean | 10 / 1e-3 | 73,728 | 10,000 |
| `pilot_mlp3_small.yaml` | mlp3_small | 14,685,184 | Adam, no schedule | none | sum | 5 / 0 | 128 | null |

If the `reconstructor` key is missing or misspelt, the code falls back to the pilot
model with no warning (F2).

## E. Readiness for full cached-feature training: NO

1. **Disk space.** The full cache needs 697.92 GB. E: had 617.67 GB free when last
   measured. `cache_features.py` stops cleanly at the 40 GB margin.
2. **F1.** With the current 10k cache, the full-COCO config passes the completeness
   check and would train on 10,000 images.
3. **Time.** The cache-fed loop measured about 157 min per epoch, so roughly 78 h for
   30 epochs. The normalisation-stats pass and the p5/p95 pass each add one more full
   read of the cache.

**Works correctly:**
- **Image ordering:** the cache uses a sorted image list, so the current 10k files are the first 10k of the full cache.
- **Pinned validation images:** the 500 pinned images are checked against the manifest.
- **Step count:** steps per epoch from the loader (878) equal the schedule's count (878) with `files_per_shard: 128`.
- **Checkpoints:** saves are atomic, and resuming from `last.pt` reproduces an uninterrupted run exactly (tested).

### Current cache, as found

| | Value |
|---|---|
| `data/cache/coco_train/features_*.pt` | 10,000 files |
| `manifest.json` | `total_images` 10000, `cached_ok` 10000, **no `complete` key** (older format) |
| First image | `features_000000.pt` = `000000000009.jpg` |
| Last image | `features_009999.pt` = `000000048794.jpg` |

## Tests

`pytest tests` gives **128 passed in 22 s**. The run used `-p no:cacheprovider`, no
bytecode, and a scratchpad temp folder, so no project files were written.

**Covered:**
- **Model:** parameter counts, a layer-by-layer match and weight-loading check against a copy of the source model, default init, forward pass, save/load.
- **Training settings:** the schedule against the source's own `SequentialLR`, clipping, loss reduction, min_delta and patience.
- **Resume:** exact resume from `last.pt`, older checkpoint formats, weights-only resume.
- **Cache and split:** the completeness check (with a manifest that has the `complete` key), the image-level split, and the pinned validation images.

**Not covered:**
- turning each YAML config into a model and asserting its depth and parameter count
- a legacy manifest without `complete` (F1)
- a mismatch between `files_per_shard` and `batch_size` (F3)
- the schedule after a weights-only resume (F4)

## F. Bugs and risks

| # | Severity | Location | Problem |
|---|---|---|---|
| F1 | **High** | `src/loss_estimator.py:721` | The code computes `m.get("complete", ok == total)`. A legacy manifest (no `complete` key, 10000/10000) therefore counts as complete. **Tested on the real cache: it passes.** `train_recon.py --config configs/base_llava_vizwiz.yaml` would train the "full COCO" run on 10k images. Nothing checks that 118,287 images are present when `max_images` is null. |
| F2 | Medium | `scripts/train_recon.py:174, 341`; `scripts/run_diagnostic.py:319`; `src/loss_estimator.py:95` | Default depth is `"mlp3"`, which maps to `mlp3_small`. A missing or misspelt key builds the pilot model, and no warning is logged. |
| F3 | Medium | `src/loss_estimator.py:885` | The step count assumes `files_per_shard`·576 is a multiple of `batch_size`. With 64: the loader runs 1,756 steps against the schedule's 878, so the cosine reaches its floor halfway through. Fine with 128, but nothing guards it. |
| F4 | Medium | `src/loss_estimator.py:938-942` | Resuming from weights only (`best.pt`) restarts the schedule at step 0, so warm-up runs again. Only the optimiser reset is logged. |
| F5 | Medium (time) | `src/loss_estimator.py:879` | Normalisation stats are not stored in `last.pt`, so every resume repeats a full pass over the cache (≈1.5–2.5 h on full COCO). |
| F6 | Low–Med | `src/loss_estimator.py:1020` | `norm_stats.pt` is written only at the end. After a crash, `best.pt` exists but the diagnostic cannot use it. |
| F7 | Low | `src/loss_estimator.py:914-937` | A resume does not check that the config, dataset size, total steps or split match the checkpoint. Only the model shape is checked, by the strict load. |
| F8 | Low | `src/loss_estimator.py:140` | The middle width follows `hidden_dim`, which is not checked: 4096 gives 37,774,336 params and 1024 gives 22,036,480, still labelled `mlp3_paper`. |
| F9 | Low | `src/loss_estimator.py:629-656` | Trainer defaults are the pilot's (Adam, no schedule, sum loss, patience 5). A missing key falls back silently. All target configs set every key explicitly. |
| F10 | Low | `scripts/train_recon.py:206` | A failed sanity check only logs a warning, and the script still exits 0. |
| F11 | Low | `src/loss_estimator.py:1008` | If a resumed run never writes `best.pt` in its own folder (for example, `last.pt` copied into a new folder), it crashes at the end. |
| F12 | Info | `src/loss_estimator.py:1086-1102` | No AMP and no gradient accumulation (see C). |

## G. Recommended fixes, in priority order

1. **F1:** when `require_complete_cache` is set and `max_images` is null, require `manifest["complete"] is True` and `n_files == 118,287` (or an expected count set in the config). Add a test with a legacy manifest.
2. **F2:** make `loss.reconstructor` required (or log a loud warning on fallback). Add a config → model test that asserts 27,282,432 parameters.
3. **F3:** count steps from the shard layout, or assert that `files_per_shard`·576 is a multiple of `batch_size`.
4. **F5 / F6:** save mean/std to `last.pt`, and write `norm_stats.pt` (with provisional p5/p95) right after the stats pass. Reuse them on resume.
5. **F4 / F7:** on a weights-only resume with a schedule, either refuse it or fast-forward the schedule. Store total steps and a config/split hash in `last.pt` and check them on resume.
6. **F8:** assert `hidden_dim == 2048` (or the parameter count) for `mlp3_paper`.
7. **F10:** exit with a non-zero code when the sanity check fails.
8. **Optional, for a closer match to the paper:** add AMP and 4-step gradient accumulation as config options, and choose between the source's actual lr curve and the intended one.

---

## Fix status (2026-11-14, after the audit)

Each finding was re-checked against the code before it was fixed. No training was
run, no cache was built, and nothing was deleted, moved or committed.
**Tests:** 187 passed (128 existing + 59 new in `tests/test_audit_fixes.py`), 0 skipped.
Output is in `results/test_report_audit_fixes.txt`.

| # | Status | Change |
|---|---|---|
| F1 | **Fixed** | `src/loss_estimator.py` `_check_cache_complete`: a manifest without `"complete": true` is refused, and the cache must cover the images the run needs. `fit()`: `max_images: null` + `require_complete_cache` needs `data.expected_images`, and the file count must equal it. `configs/base_llava_vizwiz.yaml`: `expected_images: 118287`. Checked on the real 10k cache: refused, both by the manifest check and by the count check. |
| F2 | **Fixed** | New `build_recon_net_from_config()` (`src/loss_estimator.py`), used by `train_recon.py` (model + sanity check) and `run_diagnostic.py`. `loss.reconstructor` is required (a misspelt key is named in the error); `loss.expected_params` must match. All three configs pin their size (27,282,432 / 27,282,432 / 14,685,184). |
| F3 | **Fixed** | `ShardedPatchLoader.num_batches()` counts batches per shard; the schedule uses it. A warning is logged if `files_per_shard × 576` is not a multiple of `batch_size`, and another after any epoch whose step count differs from the schedule's. |
| F4 | **Fixed** | `_make_scheduler(..., last_step)`. A weights-only resume, or a full checkpoint without scheduler state, continues the schedule at `(start_epoch−1) × steps/epoch`; warm-up is not replayed. |
| F5 | **Fixed** | The mean/std are saved to `train_stats.pt` right after the stats pass and in every `last.pt`, with a fingerprint of the train files (name, size, mtime). A resume reuses them if the fingerprint matches, otherwise it recomputes. |
| F6 | **Fixed** | Stats are on disk before epoch 1. `NormStats.save` is atomic. Resuming a checkpoint whose run already finished or early-stopped only finalises (p5/p95 → `norm_stats.pt`), with no extra epochs. p5/p95 still need the final model, so `norm_stats.pt` is not written early with wrong values. |
| F7 | **Fixed** | `last.pt` stores `run_info`. A resume refuses a different split, batch size, shard size, steps/epoch, model, parameter count, optimiser, lr, weight decay, schedule, clipping or loss reduction (override: `training.allow_resume_mismatch: true`). Changing epochs, patience or min_delta only warns. Older checkpoints without `run_info` warn. |
| F8 | **Fixed** | `mlp3_paper` with LLaVA dims and a middle width ≠ 2048 logs a warning; `loss.expected_params` makes such a config fail. Width ablations stay possible without `expected_params`. |
| F9 | **Fixed** (warning) | The trainer logs every training key that falls back to a default. The defaults themselves are unchanged, so the pilot can still be reproduced. All three configs set every key (tested). |
| F10 | **Fixed** | `sanity_check()` returns True/False; `train_recon.py` exits 3 on failure, and 2 when it refuses a config, cache or resume. |
| F11 | **Fixed** | A full resume into a folder without `best.pt` fails before training. A weights-only resume saves the resumed weights as `best.pt` if none exists. A non-finite loss stops the run, leaving `last.pt` at the last good epoch. A clear error is raised if `best.pt` is missing at the end. |
| F12 | Not changed (by design) | See below. |

Other change: `configs/pilot_mlp3_small.yaml` now has `max_images: 10000`,
`subset_mode: first` (it was `null` = all cached). Once the full cache exists,
`null` would have run the pilot config on 118k images. This setting selects the
same files with the same seed, so the pilot split is unchanged (tested).

One existing test was updated to the new F1 rule:
`tests/test_cache_and_split.py::test_require_complete_cache` now passes
`expected_images`, and also checks that leaving it out is refused.

### Differences from the paper: intentional, not bugs (F12)

| Item | Bug? | Implication | Recommendation |
|---|---|---|---|
| AMP (source) vs fp32 (ours) | No | fp32 is the more precise choice. The cache-fed loop is limited by disk (≈157 min/epoch against ≈24 min of g_φ compute, measured), so AMP would save little. | Keep fp32 and state it in Ch. 9. |
| Gradient accumulation ×4 (source) | No | The source takes one step per 4 micro-batches. We take 4× more optimiser steps of 128 images. Adding accumulation would cut steps from 878 to ~220 per epoch, so 2000 warm-up steps would be ~30 % of the run. | Keep and document. Make it a config option only if a closer replication is needed. |
| 95/5 split (ours) vs 80/20 | No | More training images. Validation is 5,914 images (~3.4M patches) and keeps the pilot's 500 pinned images. Both split by image. | Keep. |
| LR schedule | No (ours follows the intended design) | In the source, the cosine length counts micro-batches but the scheduler steps per optimiser step, so the lr decays only ~¼ of the way (code reading). Ours decays fully to 0.01·lr. | Keep, and state the difference in the report. |
