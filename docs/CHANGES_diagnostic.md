# Stage-1 diagnostic fixes

Project F26-188 (LADP). Branch `feat/faryal/stage1-diagnostic` (built on
`feat/faryal/mlp-27.3-reconstruction-module`). Findings are from
`docs/AUDIT_full_codebase.md`. The diagnostic has **not been run** yet: it needs
LLaVA-1.5-7B on the GPU (about 1 h for the 1,000 tuning examples).

## Fixed and tested

| # | Problem (before) | Fix | Evidence |
|---|---|---|---|
| C1 | The hook took E from CLIP's **last** layer (`last_hidden_state`); the cache and g_φ use `hidden_states[-2]`. Pilot loss rose from 0.085 to 0.333 on the same image. | `src/feature_capture.py` hooks the **projector**: its input is E, its output is T, exactly as LLaVA computes them | `tests/test_feature_capture.py`: on real weights, E and T are bit-identical to `features_000000.pt` |
| C2 | Crash on the first example: VizWiz answers are `{"answer": …}` dicts | `vqa_eval.answer_strings` extracts the strings and rejects malformed entries | `tests/test_diagnostic.py` (real-shaped and real VizWiz annotations) |
| H1 | Prompt without the "respond with 'Unanswerable'" instruction, and `max_new_tokens` 50 | The same `PROMPT` and `GENERATION` as the baseline (`src/vqa_eval.py`, shared) | Test checks prompt, decoding settings, fp16 cast and decoding of new tokens only |
| H2 | Simplified metric (no leave-one-out; different normalisation) labelled "VQA accuracy" | Official `vqa_acc` shared with the baseline; `any_match` kept as a separate, clearly named label; the main correlation uses the VQA error (1 − accuracy) | Leave-one-out values for 0–10 matches; all 4,319 stored baseline scores reproduce exactly (tuning accuracy 0.528184) |
| M3 | Paths resolved against the working directory | `src/config_paths.py`: config paths resolve against the project root; command-line paths against the shell | Test from another working directory |
| M4 | Any `norm_stats.pt` was accepted with any `best.pt` | NormStats carry a weights fingerprint; a mismatched pair is refused; older files (the pilot) warn, or are refused with `--require_norm_stats_fingerprint` | Pairing tests; the pilot still loads |
| M6 | Real normaliser / priority / KNOR modules untested; `smoke_test.py` checked T only | `tests/test_real_modules.py`; `smoke_test.py` also checks E against the projector input | 30 tests |
| L2 | Correlation silently NaN for a constant outcome | `degenerate` flag and a logged warning | Test |
| — | KNOR assumed "self" ranks first (fails for duplicate images) | Self is excluded explicitly | Test with duplicates |

`scripts/run_baseline.py` now imports its prompt and metric from `src/vqa_eval.py`.
The code moved there unchanged; the 4,319 stored scores reproduce bit-for-bit.

## Optional analysis (M2)

`scripts/analyze_loss_normalisation.py` measures loss distributions on COCO train /
COCO val / VizWiz tuning (disjoint calibration and evaluation halves; never
index ≥ 3800) and compares normalisation strategies.

**Result for the pilot** (300 / 300 / 600 images): the shift is small. The stored
range clips 7.4 % of VizWiz patches to 1, against 5.6 % on COCO val, and no image is
half-saturated. Recommendation: keep the stored p5 / p95 (report §5.3.3.1). The
calibration file it writes is opt-in (`loss.norm_calibration`); no config uses it.

## Not in this branch

- Relevance-scorer fix (H3) and grounding validation: FYP-2.
- Token selection, pruned-token LLaVA inference (M1): FYP-2.
