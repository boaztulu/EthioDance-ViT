# EthioDance-ViT

PyTorch implementation and benchmarks for **spatiotemporal classification of
Ethiopian traditional dances** using Video Vision Transformers (TimeSformer).

---

## Dataset

1,806 video clips across 5 Ethiopian dance styles:

| Class    | Clips |
| -------- | ----- |
| Amhara   | 516   |
| Woliyta  | 337   |
| Oromo    | 334   |
| Gurageya | 316   |
| Tigriga  | 303   |

The raw dataset is stored **outside** the repository (it is too large for git).
By default the code expects it at `../Data/ALL/<class-folder>/*.mp4` relative to
this repo; override with `data.root` in the YAML config.

## Layout

```
EthioDance-ViT/                       # <-- this repo (code only)
├── configs/                          # YAML configs (local / hipergator / base)
├── src/
│   ├── data/                         # dataset, video transforms, samplers, mixup
│   ├── models/                       # TimeSformer builder (HF transfer learning)
│   ├── losses/                       # class-balanced focal loss
│   ├── engine/                       # trainer, early stopping
│   ├── utils/                        # config, seed, metrics, checkpoint, signals
│   └── visualization/                # figure generators for the paper
├── scripts/
│   ├── prepare_splits.py             # stratified train/val/test split JSON
│   ├── train.py                      # entry point (argparse + YAML)
│   ├── visualize.py                  # paper figures from a checkpoint
│   └── train_hipergator.sh           # SLURM launcher w/ SIGUSR1 requeue
├── notebooks/
├── tests/
└── requirements.txt

../Data/ALL/                          # <-- dataset (not in git)
../experiments/<run-id>/              # <-- checkpoints, figures, logs (not in git)
../slurm_logs/                        # <-- SLURM stdout/stderr (not in git)
```

## Quick start (local smoke test)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Build splits JSON (stratified 70/15/15)
python scripts/prepare_splits.py --data-root ../Data/ALL --out configs/splits.json

# 3. Debug-mode training: 2 batches per epoch, 2 epochs, 8 frames
python scripts/train.py --config configs/local.yaml --debug

# 4. Generate paper figures from a trained checkpoint
python scripts/visualize.py --config configs/local.yaml \
    --checkpoint ../experiments/<run-id>/best.pth
```

## HiPerGator

```bash
sbatch scripts/train_hipergator.sh           # full run, requeue-safe
sbatch scripts/train_hipergator.sh --ablation
```

## Class-imbalance strategy

We combine three complementary techniques (all toggleable in YAML):

1. **Class-Balanced Focal Loss** (Cui et al., CVPR 2019) — effective-number
   reweighting with a focusing term that down-weights easy majority examples.
2. **Weighted random sampler** — produces batches that are class-balanced in
   expectation, so every batch trains every class.
3. **Video MixUp** — clip-level linear interpolation to regularize the
   overrepresented Amhara class without discarding data.

## Citation

_(to be added upon publication)_
