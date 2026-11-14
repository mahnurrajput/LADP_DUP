# Full codebase audit: LADP (F26-188)

Date: 2026-11-15. Branch audited: `feature/mlp3-paper` (with uncommitted/staged
work). `main` = `origin/main` = `f3f8523`.

**Read-only audit.** No code was changed and no Git operation was run. No training
and no cache generation. This report is the only file written.

## Summary

| Area | Verdict |
|---|---|
| Feature caching, g_φ training, resume, checkpoints | **Safe**: tested (187 tests pass) and audited earlier (F1–F11, `docs/AUDIT_mlp3_paper.md`) |
| Stage-1 diagnostic (`scripts/run_diagnostic.py`) | **Not safe to run**: 2 critical bugs (C1, C2), both confirmed on real data |
| Relevance / normaliser / priority modules (FYP-2) | **Not validated**: relevance computation is questionable (H3); the real modules have no tests |
| Existing results (baseline 0.528, 10k cache, pilot g_φ) | **Not affected** by any finding |

## What was checked

| Check | How |
|---|---|
| All source modules, scripts and configs | Read in full |
| Diagnostic E/T capture vs. training features | Ran CLIP tower + projector on cached image 0 (real weights, nothing written) |
| Diagnostic metric vs. VizWiz data | Loaded `data/vizwiz/val.json` and called the diagnostic's metric |
| Normaliser, priority module, KNOR, correlation | Called in memory with real `norm_stats.pt` and edge-case inputs |
| Both architectures | Built from configs; cross-loaded checkpoints |
| Test suite | `pytest tests`: **187 passed**, 0 skipped |
| Package versions | `requirements.txt` matches the installed env (torch 2.11.0+cu128, transformers 4.46.3, scipy 1.15.3, numpy 1.26.4, open_clip 4.2.2) |

## Execution flow as implemented

```
COCO/VizWiz images
  → scripts/cache_features.py   (VisionProjector or LlavaAdapter → E = hidden_states[-2][:,1:], T = projector(E))
  → data/cache/<split>/features_XXXXXX.pt + manifest.json
  → scripts/train_recon.py      (ReconstructionTrainer → g_φ, NormStats p5/p95)
  → checkpoints/<run_name>/{best.pt, last.pt, norm_stats.pt, train_stats.pt}
  → scripts/run_diagnostic.py   (LLaVA generate + hooks → per-patch loss, KNOR, loss–error correlation)
```

**Not implemented:** token selection (HT / SR / HRR), LLaVA inference on the
selected tokens, and the final VQA evaluation with LADP. The config keys
`strategy` and `top_k` are read by no code (see M1).

---

## Findings

### Critical

#### C1. Diagnostic captures E from the wrong CLIP layer

| | |
|---|---|
| **Where** | `scripts/run_diagnostic.py`, `_FeatureCapture._hook_vision_tower` (line 265) |
| **Problem** | The hook uses `out.last_hidden_state` (CLIP's final layer, 24). LLaVA, the cache and g_φ use `hidden_states[-2]` (layer 23). T is captured correctly from the projector, so E and T come from different layers. |
| **When it happens** | Every diagnostic run |
| **Expected vs. actual** | Loss should compare g_φ(T) with the penultimate-layer E it was trained on; it compares it with the final layer instead |
| **Fix** | Capture `out.hidden_states[config.vision_feature_layer][:, 1:]`, or the projector's input inside the projector hook. Add a test that compares captured E with the cache. |
| **Affects existing results?** | No (the diagnostic has never been run). It would have invalidated the Stage-1 losses and the KNOR result. |

**Reproduced** on `features_000000.pt` with the pilot model:

| Check | Result |
|---|---|
| cache E == `hidden_states[-2]` | yes |
| cache E == `last_hidden_state` | no |
| cosine between the two layers per patch | 0.868 mean, 0.718 min |
| cache T == `projector(hidden_states[-2])` | yes |
| mean patch loss, correct E | **0.0849** |
| mean patch loss, hook's E | **0.3334** (train p95 = 0.1430, so normalised loss ≈ 1 everywhere) |

#### C2. Diagnostic crashes on the first example

| | |
|---|---|
| **Where** | `scripts/run_diagnostic.py`, `vizwiz_accuracy` / `_normalise_answer` (lines 155–176), called from `main` (line 602) |
| **Problem** | VizWiz `answers` are dicts (`{'answer': 'unanswerable', 'answer_confidence': 'yes'}`), but the function calls `.lower()` on each entry |
| **When it happens** | First example, after LLaVA has loaded (about 1 min in) |
| **Expected vs. actual** | A score per example; instead `AttributeError: 'dict' object has no attribute 'lower'` (reproduced) |
| **Fix** | Pass `[a["answer"] for a in ann["answers"]]`, as `run_baseline.py` does |
| **Affects existing results?** | No |

### High

#### H1. Diagnostic prompt and generation differ from the baseline

| | |
|---|---|
| **Where** | `scripts/run_diagnostic.py`, `llava_answer` (line 210) |
| **Problem** | The prompt drops the baseline's instruction "When the provided information is insufficient, respond with 'Unanswerable'", and uses `max_new_tokens=50` instead of 16 |
| **Why it matters** | **448 of the first 1,000** VizWiz val examples are unanswerable (counted). Without that instruction, "correct" mostly measures answerability, which confounds the loss–error correlation. The accuracy cannot be compared with the baseline's 0.528. Size of the effect not measured. |
| **Fix** | Reuse the baseline's `PROMPT` and generation settings |
| **Affects existing results?** | No |

#### H2. Diagnostic metric is not the VQA metric the baseline uses

| | |
|---|---|
| **Where** | `scripts/run_diagnostic.py`, `vizwiz_accuracy`, `is_correct`, `build_summary` |
| **Problem** | It uses `min(n/3, 1)` without the official leave-one-out average (3 matches gives 1.0 instead of 0.9). Its normalisation differs from the baseline's (no number words, no article removal). "Correct" means at least 1 of 10 annotators matched. The summary labels that percentage "VQA accuracy". |
| **Fix** | Use `norm` / `vqa_acc` from `run_baseline.py`. Report real VQA accuracy, and keep the binary label as a separate, clearly named field. |
| **Affects existing results?** | No |

#### H3. Relevance is computed outside CLIP's joint space

| | |
|---|---|
| **Where** | `src/relevance_scorer.py`, `CLIPRelevanceScorer.compute` |
| **Problem** | CLIP's `visual.proj` is defined for `ln_post(final-layer token)`. Here it is applied to raw penultimate-layer patch tokens, with no `ln_post`. The scores are therefore not in CLIP's image–text space, contrary to the docstring. |
| **How checked** | Code reading only; not run (needs the open_clip weights) |
| **Fix** | Use the final layer with `ln_post` then `proj`. Check the scores against the VizWiz answer-grounding masks. |
| **Affects existing results?** | No (FYP-2 component, not used yet) |

### Medium

| # | Where | Problem | Fix |
|---|---|---|---|
| M1 | Pipeline | No token selection, no pruned-token LLaVA run, no LADP VQA evaluation. `strategy` / `top_k` in the configs are read by nothing. The README marks this "Not started", so it is a gap, not a bug. | FYP-2 work |
| M2 | `src/loss_estimator.py`, `_compute_norm_stats` | p5/p95 come from the trained model's own *training* patches on COCO (in-sample, optimistic) and are applied to VizWiz (domain shift), so robust-normalised loss may saturate. Not measured on VizWiz. | Fit p5/p95 on held-out COCO validation data, or on the VizWiz tuning split; measure the saturation rate |
| M3 | `scripts/run_diagnostic.py`, `main` | `output_dir`, the checkpoint folder and the data paths resolve against the current directory. `train_recon.py` resolves against the project root. Running it from another folder fails or writes elsewhere. | Resolve against the project root |
| M4 | `scripts/run_diagnostic.py`, `load_loss_estimator` | Nothing checks that `norm_stats.pt` belongs to the given `best.pt`. Stats from another run of the same architecture load silently (a different architecture fails loudly). | Store a run ID / fingerprint in both files and compare |
| M5 | `scripts/cache_features.py`, `write_manifest` | A single unreadable image leaves `complete: false` forever, so the F1 guard blocks full-COCO training with no override | Add an explicit, logged list of accepted failures |
| M6 | `tests/` | See "Test coverage" below: the real normaliser, priority, KNOR, diagnostic, baseline and adapter code has no tests. `test_relevance_scorer.py::TestNormaliser` tests a local copy (which even has a `robust` mode the real class lacks). `scripts/smoke_test.py` checks T but not E, which is why C1 went unnoticed. | Test the real modules; add an E check to the smoke test |

### Low

| # | Where | Problem |
|---|---|---|
| L1 | `src/normaliser.py` vs. the scripts | Mixed import styles (`src.loss_estimator` vs. `loss_estimator`) create two `NormStats` classes (checked: they are not the same object). Harmless until something uses `isinstance`. |
| L2 | `src/diagnostics/knor.py`, `correlate_loss_with_error` | If all answers are correct (or all wrong) it silently returns NaN correlations (checked). |
| L3 | `scripts/run_baseline.py` | The summary crashes if nothing has been done. `n_total` counts records beyond `--limit`. Held-out per-example accuracy is stored in `predictions.jsonl` (gitignored), so it can be seen by opening the file. |
| L4 | `scripts/run_diagnostic.py` | The progress log's mean loss divides by a count that includes NaN records. |
| L5 | `scripts/cache_features.py` | The manifest's `cached` field means "reused" (`True`) vs. "newly written" (`False`), which is misleading. The adapter engine never closes `Image.open`. With fp32 precision it would re-extract every run (the dtype check expects fp16). |
| L6 | `src/diagnostics/knor.py`, `run_diagnostic.build_summary` | The KNOR interpretation thresholds (0.95 / 0.70) are not from the paper. |

---

## Both architectures: independent and not mixed

| Check | Result |
|---|---|
| `mlp3_small` parameters | 14,685,184 |
| `mlp3_paper` parameters | 27,282,432 (LayerNorms at `net.1`, `net.5`) |
| Pilot `best.pt` → `mlp3_paper` | Fails (size mismatch) |
| `mlp3_paper` weights → `mlp3_small` | Fails |
| Pilot `best.pt` → `mlp3_small` (this branch) / `mlp3` (`main`) | Loads |
| Configs | Each pins `reconstructor` and `expected_params` |
| Checkpoint folders | Separate: `recon_pilot`, `recon_paper_10k_mlp3_paper`, `recon_full_mlp3_paper` |
| Known risk | Re-running the pilot config overwrites `checkpoints/recon_pilot/` (documented in the README) |

## Test coverage

| Area | Tested? |
|---|---|
| `ReconstructionNet` (all variants), `LossEstimator`, `NormStats` | Yes |
| `ReconstructionTrainer`: split, schedule, clipping, early stopping, resume, F1–F11 | Yes |
| `ShardedPatchLoader` | Yes |
| `cache_features.py` (fake extractor): resume, verification, disk-space stop | Yes |
| `CLIPRelevanceScorer.compute` (projection mocked) | Partly; real open_clip path untested |
| `src/normaliser.py` | **No** (the tests use a local copy) |
| `src/priority_module.py` | **No** (only its `__main__` self-check) |
| `src/diagnostics/knor.py` | **No** (only its `__main__` self-check) |
| `scripts/run_diagnostic.py` | **No** (C1, C2 slipped through) |
| `scripts/run_baseline.py` (metric, resume) | **No** |
| `src/connector_adapter.py`, `src/onthefly_features.py` | No unit tests (GPU and weights needed); E/T checked manually against the cache |

## Effect on existing results

| Result | Affected? | Why |
|---|---|---|
| Baseline 0.528 (`run_baseline.py`) | No | Correct prompt, official leave-one-out metric |
| COCO 10k cache | No | Bit-identical to `hidden_states[-2]` and to the projector output |
| Pilot g_φ (val MSE 0.072404) | No | Trained on the correct cache |
| Stage-1 diagnostic | — | Never run. It would have crashed (C2); once fixed, it would have given invalid losses (C1). |

## Recommended fix order

1. **C2:** pass answer strings to the metric.
2. **C1:** capture E from `hidden_states[-2]`; add a test that compares it with the cache.
3. **H1 + H2:** reuse the baseline's prompt and metric. The diagnostic then becomes valid; add tests.
4. **M3, M4, M6:** project-root paths, a norm_stats/checkpoint match check, tests for the real modules and an E check in `smoke_test.py`.
5. **H3, M2:** before FYP-2, fix the relevance projection and the domain of the p5/p95 statistics.

Nothing has been fixed yet; fixes wait for approval.
