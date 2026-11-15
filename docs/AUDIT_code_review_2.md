# Second audit: code review of the uncommitted changes

Project F26-188 (LADP). Date: 2026-11-15.
First audits: `docs/AUDIT_mlp3_paper.md` (F1–F11, fixed) and a full-codebase
audit (C1–L6, mainly the Stage-1 diagnostic; kept outside this branch).

**Scope:** the working-tree changes against `HEAD` on `feature/mlp3-paper`:
- `src/loss_estimator.py`, `src/onthefly_features.py`
- `scripts/cache_features.py`, `scripts/train_recon.py`, `scripts/run_diagnostic.py`
- the new benchmark and split scripts

Focus: modularity, security, and edge cases.

**Method:** a code review by reading the code. Nothing was changed or run.
The findings below have **not** been reproduced by running code. The "How it
fails" column explains each one from the code. No `CLAUDE.md` exists in the
repository or at user level, so there were no project conventions to check.

## Summary

| Area | Result |
|---|---|
| Security | **No real issues found** |
| Edge cases | 4 problems (3 Medium, 1 Low) |
| Modularity | 2 problems (duplicated helpers, scattered path resolution) |
| Efficiency | 1 problem (resume checkpoint loaded twice) |
| Overlap with the full audit | 2 findings repeat M3 and L5 |

## Security

No vulnerabilities found:
- **Checkpoint loading:** every `torch.load` uses `weights_only=True`, so a crafted checkpoint cannot run code when it is loaded.
- **Model loading:** the Hugging Face snapshot is loaded with `local_files_only=True` (no network download).
- **Pinned validation split:** file names from the JSON are matched by basename against the cache file list, so the JSON cannot point to files outside the cache.

## Findings

### Medium

| # | Where | Problem | How it fails | Fix |
|---|---|---|---|---|
| R1 | `src/loss_estimator.py:1125` (`fit`, resume check) | With `training.warmup_ratio` set, `warmup_steps` is computed from `total_steps` (= epochs × steps/epoch). Changing `epochs` is meant to be warn-only (`_RESUME_SOFT`), but it changes `warmup_steps`, which must match (`_RESUME_STRICT`). This contradicts fix F7. | `configs/paper_10k_mlp3_paper.yaml` (`warmup_ratio: 0.05`, 30 × 75 steps → 112 warm-up steps). Resume `last.pt` with `epochs: 40` → warm-up becomes 150 → `ValueError: warmup_steps: checkpoint=112 now=150`, and `train_recon.py` exits 2. | Store `warmup_ratio` in `run_info` and compare that instead, or treat `warmup_steps` as warn-only when it comes from the ratio |
| R2 | `src/loss_estimator.py:1021` (`fit`, subset selection) | If `data.max_images` ≤ the number of pinned validation images, the subset is the pinned set alone: all of it goes to validation and none to training. | `max_images: 500` with the 500-image pilot split: train is empty, and the run fails with "needs at least one file" (otherwise `n_train % len(train_files)` would divide by zero). `max_images: 300` silently uses 500 images. | Check `max_images > len(fixed_val)` up front and stop with a clear message |
| R3 | `src/loss_estimator.py:1213` (weights-only resume) | The resumed weights are saved as `best.pt` only if none exists. An older `best.pt` in the run folder, possibly from a different aborted attempt, is kept and later reloaded as "best" without being checked against `--best-val`. | `--resume <other>/best.pt --start-epoch 10 --best-val 0.30` into a folder with a stale `best.pt`. If no epoch beats 0.30, the stale file is reloaded and `norm_stats.pt` is fitted on the wrong weights. | On a weights-only resume, always save the resumed weights as `best.pt` (backing up any existing one), or refuse if a `best.pt` already exists |

### Low

| # | Where | Problem | How it fails | Fix |
|---|---|---|---|---|
| R4 | `src/onthefly_features.py:50` (`VisionProjector`) | Assumes a sharded checkpoint (`model.safetensors.index.json`) and an integer `vision_feature_layer`. | A single-file `model.safetensors` snapshot raises `FileNotFoundError`, and a list-valued layer such as `[-2, -5]` raises `TypeError`. Both show only as "Failed to load the model". | Support single-file checkpoints; validate the layer type with a clear message |
| R5 | `scripts/cache_features.py:581` | The default `vision_projector` engine treats any precision other than `"fp16"` as fp32 without checking it. `LlavaAdapter` rejects anything except `fp16` / `fp32`. | `precision: bf16` or the typo `FP16` silently extracts in fp32, which then triggers R6. | Validate precision the same way for both engines |
| R6 | `scripts/cache_features.py:75` (`verify_cache_file`) | Verification requires fp16, but extraction saves in the model's dtype (fp32 when precision is not `fp16`). The script's own files fail verification on every later run. Same as full-audit **L5**. | With `precision: fp32`, every rerun re-extracts all files (118k for full COCO). | Cast to fp16 before saving, or verify against the configured dtype |
| R7 | `scripts/run_diagnostic.py:550` | Default checkpoint and NormStats paths come from a relative `checkpoints_dir` resolved against the current folder. `train_recon.py` resolves against the project root. Same as full-audit **M3**. | Running from outside the repository root looks for `<cwd>/checkpoints/<run_name>/best.pt` and fails with `FileNotFoundError`. | Resolve against the project root (see R10) |

### Efficiency

| # | Where | Problem | Effect | Fix |
|---|---|---|---|---|
| R8 | `scripts/train_recon.py:372` | The resume checkpoint is loaded once in `main()`, only to tell a full checkpoint from bare weights, and then again inside `fit()`. | For `mlp3_paper`, `last.pt` (model + Adam moments, about 330 MB) is read and deserialised twice per resume, and two copies are briefly in RAM. | Load once and pass the dict to `fit()`, or let `fit()` report the checkpoint type |

### Modularity

| # | Where | Problem | Risk | Fix |
|---|---|---|---|---|
| R9 | `scripts/cache_features.py:226` (`_CacheImages`, `atomic_torch_save`) | Duplicates helpers added elsewhere in the same changes: `ImagePathDataset` in `src/onthefly_features.py`, and the tmp-file-then-rename save in `ReconstructionTrainer._atomic_save` and `NormStats.save`. | Image loading lives in two dataset classes, and atomic saving is written three times. A fix to one copy (e.g. cleaning up the tmp file on failure, or the RGB/preprocessing path) will not reach the others, so the cache and the on-the-fly path can drift apart. | Move one tolerant `ImagePathDataset` and one `atomic_save` helper into `src/` and import them |
| R10 | `scripts/train_recon.py:356` | Config paths are resolved against the project root one key at a time (`checkpoints_dir`, `fixed_val_images`) in `train_recon.py` only. `ReconstructionTrainer` and `run_diagnostic.py` use the raw relative values. | Using the trainer directly (tests, notebooks) or running the diagnostic resolves paths against the current folder, so the pinned split or the checkpoints are missing or written to the wrong place. Every new path key needs another special case. | One `resolve_config_paths(cfg, project_root)` helper in `src/`, used by every entry point |

## Suggested tests

| Test | Covers |
|---|---|
| Resume with `warmup_ratio` set after changing `epochs` | R1 |
| `max_images` at and below the pinned validation set size | R2 |
| Weights-only resume into a folder that already holds a `best.pt` | R3 |
| Build an fp32 cache, then run again: nothing should be re-extracted | R5, R6 |
| `run_diagnostic` default paths when run from outside the repository root | R7, R10 |

## Status

| Finding | Status (2026-11-15) |
|---|---|
| R1, R3 | **Fixed and tested** (`docs/CHANGES_R1_R3.md`, `tests/test_resume_r1_r3.py`) |
| R7 | Fixed with the Stage-1 diagnostic work (separate, not in this branch) |
| R2, R4, R5, R6, R8, R9, R10 | Open (none affects the existing results or the 10k / full-COCO configs) |

Suggested order for the open items: R2 (training edge case), then R5 + R6 (cache
precision), then R10 (one shared path helper), then R8, R9 and R4.
