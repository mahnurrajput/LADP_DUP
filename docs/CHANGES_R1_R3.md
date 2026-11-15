# Resume fixes R1 and R3

Project F26-188 (LADP). Date: 2026-11-15. Findings from `docs/AUDIT_code_review_2.md`.
Both were reproduced before they were fixed. Only `src/loss_estimator.py` changed;
the regression tests are in `tests/test_resume_r1_r3.py` (16 tests).

## R1: extending epochs on resume

**Root cause:**
1. With `training.warmup_ratio`, `warmup_steps` was re-derived from the new
   `total_steps`, and that derived value had to match the checkpoint. A resume with
   more epochs was therefore refused (`warmup_steps: checkpoint=6 now=12`).
2. Even with a fixed `warmup_steps`, the cosine was recomputed over the new total,
   so the learning rate jumped **up** at the resume point.

**Fix:**
- **Warm-up frozen:** on resume, the warm-up stays at the checkpoint's value
  (those steps are already done).
- **Cosine re-based:** with a different total, the cosine restarts from the
  checkpoint's current learning rate and reaches `min_lr_ratio · lr` at the new end,
  so the learning rate is continuous at the resume point.
- **Schedule saved:** the schedule is stored in `last.pt` (`schedule`), so repeated
  extensions chain correctly.
- **What is compared:** the resume check uses the *configured* warm-up
  (`warmup_config`: ratio or steps), not the derived step count. Changing the
  warm-up is still refused, and so is shortening a run below the steps already taken.
- **Unchanged epochs:** resuming with the same `epochs` is exactly as before
  (bit-identical to an uninterrupted run).

## R3: stale `best.pt` on resume

**Finding:** on a weights-only resume, `best.pt` was written from the resumed
weights only if it was missing. An existing (possibly stale) `best.pt` was kept and
reloaded at the end as "best", and `norm_stats.pt` was then fitted on it.

**Root cause:** nothing recorded *which* weights `best.pt` should hold, and the final
reload trusted whatever file was on disk.

**Fix:**
- **Fingerprint recorded:** `last.pt` stores `best_sha256`, a fingerprint of the
  weights in `best.pt`, updated at every new best.
- **Weights-only resume:** the resumed weights are, by `--best-val`, the best so far,
  so they become `best.pt`. A different existing `best.pt` is kept as
  `best.superseded_<time>.pt`, never deleted.
- **Full resume:** a `best.pt` whose fingerprint does not match is refused before
  any training.
- **Final reload:** checked against the fingerprint before `norm_stats.pt` is fitted.
- **Older checkpoints** without the fingerprint (e.g. the pilot's) still resume,
  with a warning.
