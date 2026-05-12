# CauAir

Official implementation of **CauAir: Causal Learning Meet Covariates for Nationwide Air Quality Forecasting**.

## LargeAQ Download

LargeAQ is the core contribution of this project. If you are here to reproduce the main paper result, start from the dataset release:

- **LargeAQ dataset**: https://drive.google.com/file/d/1fBfa4fek4OPZlC-jufs11ocHK3KRXGdX/view?usp=sharing

## This Update

This repository update mainly adds and organizes:

1. **Baseline code**
2. **Other related components**

Concretely, the current version now includes the maintained baseline experiment entrypoints, shared training engines, supporting utilities, and a cleaner repository structure for reproduction and follow-up development.

## Overview

CauAir studies nationwide air-quality forecasting with explicit covariate modeling and lightweight causal structure. This repository now contains:

- the official CauAir implementation
- baseline experiment code
- related training and evaluation components
- processed KnowAir and CCAQ-compatible data roots
- shared logging, path, metric, and dataloader utilities

The repository has been cleaned and reorganized around a simple rule:

- `src/models` is the canonical model inventory
- `experiments` contains runnable experiment entrypoints
- generated artifacts are separated from source code

## Repository Structure

```text
CauAir-main/
├── experiments/         # Runnable experiment entrypoints
├── src/
│   ├── base/            # Base model / engine abstractions
│   ├── engines/         # Training and evaluation loops
│   ├── models/          # Canonical model implementations
│   └── utils/           # Data loading, metrics, logging, project paths
├── data/                # LargeAQ-style local data root
├── data_knowair/        # KnowAir data root
├── datagagnn/           # CCAQ / GAGNN-style data root
├── outputs/             # Logs and checkpoints, created on demand
└── requirements.txt
```

## Datasets

### LargeAQ

LargeAQ is the nationwide, long-term air-quality dataset introduced for CauAir.

Expected local layout:

```text
data/
└── 24_24/
    └── all/
        ├── his.npz
        ├── idx_train.npy
        ├── idx_val.npy
        └── idx_test.npy
```

### KnowAir

This repository already includes processed KnowAir data under:

```text
data_knowair/
├── 24_24/
│   └── all/
└── adj_mx.npy
```

### CCAQ

This repository already includes processed CCAQ / GAGNN-style data under:

```text
datagagnn/
├── 24_24/
│   └── all/
└── adj_mx.npy
```

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch
- NumPy / SciPy / scikit-learn / pandas
- `torch-geometric` and `torch-scatter` for graph-based baselines
- `torchdiffeq`, `torcheval`, `einops`, `reformer-pytorch`, and other model-specific utilities

For graph libraries, it is usually better to install PyTorch first and then install the matching PyG packages for your CUDA or CPU environment.

## Running Experiments

All maintained experiment entrypoints are under `experiments/<model_name>/main.py`.

### CauAir on LargeAQ

```bash
python experiments/cauair/main.py \
  --device cuda:0 \
  --model_name cauair \
  --dataset 24_24 \
  --input_dim 8 \
  --tod 24 \
  --dim 128 \
  --head 8 \
  --rank 32
```

### CauAir on KnowAir

```bash
python experiments/cauair/main.py \
  --device cuda:0 \
  --model_name cauair \
  --dataset 24_24_KA \
  --input_dim 13 \
  --tod 8 \
  --dim 128 \
  --head 2 \
  --rank 10
```

### CauAir on CCAQ

```bash
python experiments/cauair/main.py \
  --device cuda:0 \
  --model_name cauair \
  --dataset 24_24_G \
  --input_dim 10 \
  --tod 24 \
  --dim 128 \
  --head 4 \
  --rank 108
```

## Outputs

Experiment scripts may still pass legacy paths such as `./experiments/<model>/<dataset>/`, but generated artifacts are normalized to:

```text
outputs/experiments/<model>/<dataset>/
```

This keeps logs and checkpoints out of the source tree while preserving compatibility with older scripts.

## Notes

The current repository version focuses on putting the usable codebase online in a cleaner form. In particular:

- baseline code is now included in the repository
- other related components needed for running and comparing models are also included
- empty experiment artifact folders and obsolete residual branches have been removed

## Citation

If this repository or the LargeAQ dataset helps your work, please cite:

```bibtex
@article{ma2025causal,
  title={Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting},
  author={Ma, Jiaming and Cui, Zhiqing and Wang, Binwu and Wang, Pengkun and Zhou, Zhengyang and Zhao, Zhe and Wang, Yang},
  journal={International Joint Conference on Artificial Intelligence},
  year={2025}
}
```
