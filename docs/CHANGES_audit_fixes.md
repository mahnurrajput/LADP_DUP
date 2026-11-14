# Changes made to fix the implementation audit

Project F26-188 (LADP). Date: 2026-11-14.
Audit report: `docs/AUDIT_mlp3_paper.md`.

**Not done:** no training, no feature-cache generation, no Git commit or push.
Nothing was deleted or moved.

**Backups:** copies of every edited file, taken before editing, are in the session
scratchpad (`…\scratchpad\pre_audit_fix_backup\`). The scratchpad is temporary and
can be cleared; copy the folder somewhere permanent if you want to keep it.

## Summary

| Item | Result |
|---|---|
| Audit issues fixed | F1–F11 (all confirmed in the code first) |
| Not changed | F12, the paper differences (by design, see the last section) |
| `mlp3_paper` parameters | 27,282,432 (unchanged) |
| 10k cache passes as full COCO? | No, refused |
| Tests | 187 passed, 0 skipped (128 existing + 59 new) |

## Files changed

| File | Type | What changed |
|---|---|---|
| `src/loss_estimator.py` | code | Main fixes, F1–F11 (details below) |
| `scripts/train_recon.py` | code | Uses the new model builder; returns proper exit codes |
| `scripts/run_diagnostic.py` | code | Uses the new model builder (no silent fallback to the pilot model) |
| `configs/base_llava_vizwiz.yaml` | config | Added `data.expected_images: 118287` and `loss.expected_params: 27282432` |
| `configs/paper_10k_mlp3_paper.yaml` | config | Added `loss.expected_params: 27282432` |
| `configs/pilot_mlp3_small.yaml` | config | Added `loss.expected_params: 14685184`; `max_images: null` → `10000` with `subset_mode: first` |
| `tests/test_cache_and_split.py` | test | `test_require_complete_cache` updated to the new F1 rule |
| `tests/test_audit_fixes.py` | test (new) | 59 tests for the fixes |
| `README.md` | docs | Safety-check table, status, test count |
| `docs/AUDIT_mlp3_paper.md` | docs | "Fix status" section added |
| `results/test_report_audit_fixes.txt` | report (new) | Full test output |

## Code changes by issue

### F1: the 10k cache could pass as the full COCO cache (critical)

**Before:** a manifest without a `"complete"` key counted as complete whenever
`cached_ok == total_images`. A run set to "use all images" also never checked how
many images it should have. So the full-COCO config would have trained on the 10k
cache with no error.

**Now** (`src/loss_estimator.py`):
- **`_check_cache_complete(cache_dir, n_files, required_images)`:**
  - a manifest without `"complete": true` is refused, with a message to re-run `cache_features.py`;
  - `cached_ok` must equal `total_images`;
  - the manifest and the files on disk must cover at least `required_images`.
- **`fit()`:**
  - with `max_images: null` and `require_complete_cache: true`, `data.expected_images` must be set, or the run stops;
  - with `max_images: null`, the number of cache files must equal `expected_images`;
  - all checks run before anything is written to disk.

**Config:** `data.expected_images: 118287` in `base_llava_vizwiz.yaml`.

**Checked on the real 10k cache:** refused twice, once for the missing flag and once
for the count ("…complete for 10000 images … needs 118287").

### F2: silent fallback to `mlp3_small`

**Before:** a missing or misspelt `loss.reconstructor` built the pilot model with no
message.

**Now:**
- New function `build_recon_net_from_config(config)`:
  - `loss.reconstructor` is required, and a misspelt key is named in the error;
  - `loss.expected_params`, if set, must equal the model's parameter count.
- Used by `train_recon.py` (model and sanity check) and `run_diagnostic.py`.
- `train_recon.py` exits with code 2 on a bad model config.

### F3: wrong step count for the learning-rate schedule

**Before:** steps per epoch were computed as `ceil(total_patches / batch_size)`.
Batches never cross shard boundaries, so with some shard sizes the schedule would end
early.

**Now:**
- New `ShardedPatchLoader.num_batches(patches_per_file)` gives the exact count; the schedule uses it.
- A warning is logged if `files_per_shard × 576` is not a multiple of `batch_size`.
- A warning is logged if an epoch runs a different number of steps from the schedule's.

### F4: weights-only resume restarted warm-up

**Before:** resuming from `best.pt`, or from an old `last.pt` without scheduler state,
started the schedule again at step 0.

**Now:** `_make_scheduler(..., last_step=k)` starts the schedule at step
`(start_epoch − 1) × steps_per_epoch`, so warm-up is not repeated.

### F5: normalisation stats recomputed on every resume

**Before:** every resume repeated a full pass over the cache to recompute mean/std.

**Now:**
- Mean/std are saved to `<checkpoint folder>/train_stats.pt` and inside every `last.pt`.
- They are stored with a fingerprint of the train files (name, size, modified time).
- On resume they are reused if the fingerprint matches, otherwise recomputed.

### F6: statistics could be lost after a crash

**Now:**
- **Before epoch 1:** stats are already on disk.
- **`NormStats.save`:** writes a temp file, then renames it (atomic).
- **Finished runs:** resuming a run that already finished or early-stopped only finalises it (p5/p95 and `norm_stats.pt`), with no extra epochs.
- **p5/p95:** still computed only from the final model, so `norm_stats.pt` is never written with wrong values.

### F7: resume did not check the settings

**Now:** `last.pt` stores `run_info`, the settings the run used.
- **Resume refuses** a checkpoint whose split, number of images or patches, batch size,
  shard size, steps per epoch, model, parameter count, optimiser, lr, weight decay,
  schedule, warm-up, clipping or loss reduction differs from the current run.
- **Changing epochs, patience or min_delta** only logs a warning.
- **Old checkpoints** without `run_info` (like the pilot's) log a warning and still work.
- **Override:** `training.allow_resume_mismatch: true`.

### F8: `mlp3_paper` width not checked

**Now:**
- `mlp3_paper` with LLaVA dimensions and a middle width other than 2048 logs a warning.
- With `loss.expected_params` set, such a config fails.
- Width ablations still work if `expected_params` is not set.

### F9: silent trainer defaults

**Now:** the trainer logs every training setting that falls back to a default. The
defaults themselves are unchanged, so the pilot can still be reproduced. All three
configs set every key (tested).

### F10: a failed sanity check still exited 0

**Now:** `sanity_check()` returns True or False.

| Exit code | Meaning |
|---|---|
| 3 | the sanity check failed, or `best.pt` / `norm_stats.pt` is missing at the end |
| 2 | a config, cache or resume was refused, or training stopped with an error |

### F11: missing `best.pt` and non-finite loss

**Now:**
- **Full resume into a folder without `best.pt`:** fails before training starts.
- **Weights-only resume with no `best.pt` in the folder:** the resumed weights are saved as `best.pt`.
- **NaN or inf loss:** the run stops, and `last.pt` stays at the last good epoch.
- **`best.pt` missing at the end:** a clear error is raised.

## Config change to the pilot

`configs/pilot_mlp3_small.yaml`:

```yaml
# before
max_images: null      # = all cached images
# after
max_images: 10000
subset_mode: first
```

**Why:** once the full cache exists, "all cached images" would have run the pilot
config on 118k images. The new setting selects the same 10,000 files with the same
seed, so the pilot split is unchanged (tested).

## Test changes

**Updated:** `tests/test_cache_and_split.py::test_require_complete_cache`. It now
passes `expected_images`, and also checks that leaving it out is refused. The test
changed because the F1 rule changed on purpose.

**New:** `tests/test_audit_fixes.py`, 59 tests:

| Area | What is tested |
|---|---|
| F1 | Legacy, incomplete and too-small caches are refused; refusal happens before anything is written; the real 10k cache is refused by the real full-COCO config |
| F2 / F8 | Missing or misspelt reconstructor key; `expected_params` mismatch; real configs build exactly 27,282,432 / 27,282,432 / 14,685,184 parameters; layer structure unchanged; the training script exits 2 |
| F3 | Batch count equals the real loader for 6 layouts; full-COCO layout = 878 steps; the schedule reaches its minimum exactly on the last step |
| F4 | Weights-only and scheduler-less resumes match an uninterrupted run's schedule; exact resume is still bit-identical |
| F5 / F6 | Stats reused on resume; saved before epoch 1; recomputed when a file changes; finalise-only resume; early-stopped run gets no extra epochs; atomic save |
| F7 | Six kinds of mismatch refused; model change refused; override flag; changing epochs only warns; old checkpoint warns |
| F9 / F10 / F11 | Default warnings; sanity check returns False; missing `best.pt`; non-finite loss |
| Pilot | Pilot checkpoint loads through both the trainer path and the diagnostic loader (p5/p95 = 0.020985 / 0.142976); pilot split unchanged |

**Result:** `python -m pytest tests` → 187 passed, 0 skipped
(`results/test_report_audit_fixes.txt`).

## Not changed

**Kept as they were:**
- the `mlp3_paper` layer structure;
- the reconstruction objective;
- the frozen CLIP encoder and projector;
- cached-feature training (no switch to on-the-fly extraction);
- the image-level split and the 500 pinned validation images;
- all datasets, caches and checkpoints.

**Paper differences (F12), kept on purpose:**

| Item | Why kept |
|---|---|
| fp32 instead of AMP | Training is limited by disk speed, so AMP would save little; fp32 is more precise |
| No gradient accumulation (source: ×4) | Adding it would cut steps to ~220 per epoch, and the 2000-step warm-up would become ~30 % of the run |
| 95/5 split instead of 80/20 | More training data; keeps the 500 pinned pilot images; both split by image |
| Full cosine decay | Follows the intended schedule; in the source code the lr only decays about ¼ of the way |

All four should be stated in Chapter 9 of the report.
