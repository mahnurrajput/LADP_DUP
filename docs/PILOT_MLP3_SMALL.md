# FYP-1 pilot reconstruction module: `mlp3` (14,685,184 parameters)

Project F26-188 (LADP). Branch `feat/faryal/mlp-small-reconstruction-module`.

This branch holds **exactly the code that trained the FYP-1 pilot g_φ**:
`src/loss_estimator.py` and `scripts/train_recon.py` as in commit `9a488d8`
("Add resume support to reconstruction training"). Nothing newer is included:
not the 27.3M `mlp3_paper` model, the AdamW / warm-up + cosine options, the
audit fixes F1–F11, or the resume fixes R1/R3. Those are on
`feat/faryal/mlp-27.3-reconstruction-module`.

## Model

g_φ maps a post-projector LLaVA token T (4096) to the pre-projector CLIP feature E (1024),
in standardised space:

```
Linear(4096 → 2048) → GELU → Dropout(0.1) → Linear(2048 → 2048) → GELU → Dropout(0.1) → Linear(2048 → 1024)
```

| | |
|---|---|
| Name in code | `mlp3` (called `mlp3_small` on the newer branch; same network and weights layout) |
| Parameters | 14,685,184 |
| Initialisation | Kaiming uniform, zero bias |
| Input / output | (576, 4096) → (576, 1024) per image |

## Data

| | |
|---|---|
| Features | `data/cache/coco_train/features_000000.pt` … `features_009999.pt`: first 10,000 sorted COCO train2017 images. E = CLIP ViT-L/14-336 `hidden_states[-2]` without CLS (576 × 1024), T = LLaVA-1.5 projector output (576 × 4096), fp16 |
| Split | By image: `randperm(10000, seed 42)`, 5 % validation = **500 images**, listed in `configs/splits/coco_pilot_val500.json` (written by `scripts/export_pilot_val_split.py` from the manifest; it only reads file names) |
| Normalisation | Train-split mean / std (streaming); robust p5 / p95 of per-patch train loss |

## Configuration

`configs/base_llava_vizwiz.yaml` on this branch is **identical** to the config the pilot
recorded (`results/pilot_mlp3_small/config_snapshot.yaml`; checked key by key):

| Key | Value |
|---|---|
| `loss.reconstructor` / `hidden_dim` / `dropout` | `mlp3` / 2048 / 0.1 |
| `training.lr` / `batch_size` / `epochs` | 1e-4 / 128 patches / 30 |
| `training.patience` / `val_fraction` / `seed` | 5 / 0.05 / 42 |
| Optimiser / loss | Adam, constant lr, no clipping; summed MSE in standardised space |

## Training run (14 Nov 2026)

1. Started from scratch. The process stopped without an error after epoch 7; the best checkpoint was epoch 7 (val 0.076641).
2. Resumed from the weights with:
   `python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml --resume checkpoints/recon/best.pt --start-epoch 8 --best-val 0.076641`.
   The optimiser state started fresh, because a weights-only resume carries none.
3. Early stopping at epoch 21 (patience 5). **Best epoch 16: validation MSE 0.072404.**
   Epochs 17–21 diverged (train loss 0.10 → 0.30).
4. NormStats: p5 = 0.020985, p95 = 0.142976.
5. Check: recomputing the validation MSE of `best.pt` on the 500 images gives 0.07240369 (diff 3.1e-7).

## Artefacts (not in Git; kept on disk)

| Path | Contents |
|---|---|
| `checkpoints/recon/` and the copy in `checkpoints/recon_pilot/` | `best.pt` (58,743,473 B), `last.pt`, `norm_stats.pt`, `best_epoch07_backup.pt` |
| `results/stage1_diagnostic/` and the copy in `results/pilot_mlp3_small/` | training log, loss curve, config snapshot |

`checkpoints/` and `data/` are git-ignored. The results files are left out of this branch
as generated output.

## Reproduce

```bash
# train (writes checkpoints/recon/ and results/stage1_diagnostic/ — back up the pilot first)
python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml
# resume after an interruption
python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml --resume checkpoints/recon/last.pt
python scripts/train_recon.py --config configs/base_llava_vizwiz.yaml --resume checkpoints/recon/best.pt --start-epoch N --best-val X
# tests
python -m pytest tests -q
```

> **Warning:** retraining with this config writes into `checkpoints/recon/` and
> `results/stage1_diagnostic/`, where the pilot run is stored.
