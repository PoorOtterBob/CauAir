# CauAir

Official implementation of **CauAir: Causal Learning Meet Covariates for Nationwide Air Quality Forecasting**.

This repository centers on the `src/models/cauair.py` model and keeps a cleaned benchmark suite of related forecasting baselines. The codebase has been reorganized so that:

- `src/models` is the canonical model inventory
- `experiments` contains runnable entrypoints only
- training logs and checkpoints are redirected to `outputs/`
- stale private branches and empty experiment artifact folders are removed

## Project Scope

CauAir targets long-horizon, multi-station air-quality forecasting with covariate-aware temporal modeling. The repository also includes a collection of baseline models used for comparison on nationwide air-quality datasets.

The current repository is organized around three practical goals:

1. keep model implementations under a single canonical location: `src/models`
2. make experiment entrypoints easier to audit and reproduce
3. separate source code from generated outputs such as logs and checkpoints

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
├── outputs/             # Logs and checkpoints (created on demand)
└── requirements.txt
```

## Datasets

### 1. LargeAQ

LargeAQ is the nationwide, long-term air-quality dataset introduced for CauAir.

The original README referenced an external release link. If you maintain a public release page or cloud storage link for LargeAQ, place it here explicitly before publishing the repository.

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

### 2. KnowAir

This repository includes processed KnowAir data under:

```text
data_knowair/
├── 24_24/
│   └── all/
└── adj_mx.npy
```

### 3. CCAQ

This repository includes processed CCAQ / GAGNN-style data under:

```text
datagagnn/
├── 24_24/
│   └── all/
└── adj_mx.npy
```

## Environment

Install the project dependencies with your preferred Python environment manager.

Example:

```bash
pip install -r requirements.txt
```

The repository depends on:

- PyTorch
- NumPy / SciPy / scikit-learn / pandas
- `torch-geometric` and `torch-scatter` for graph-based models
- `torchdiffeq`, `torcheval`, `einops`, `reformer-pytorch`, and a few model-specific utilities

Because several graph libraries are platform-sensitive, it is usually safer to install PyTorch first and then install the PyG-related packages against the matching CUDA / CPU build.

## Running Experiments

All runnable entrypoints are under `experiments/<model_name>/main.py`.

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

## Output Convention

Experiment scripts may still pass legacy paths like `./experiments/<model>/<dataset>/`, but the shared logging utility now normalizes them to the generated output tree:

```text
outputs/experiments/<model>/<dataset>/
```

This keeps code and generated artifacts separate while preserving backward compatibility for scripts that expect the old naming convention.

## Notes on Repository Cleanup

This repository has been cleaned to align with the canonical model list under `src/models`.

The cleanup specifically removed:

- orphaned experiment entrypoints that referenced missing models or engines
- private / ad hoc experiment branches that were no longer runnable
- empty experiment artifact folders left by earlier runs
- Finder metadata and Python cache directories

As a result, `experiments/` now represents the maintained runnable entrypoints only.

## Citation

If you use this repository in academic work, cite:

```bibtex
@article{ma2025causal,
  title={Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting},
  author={Ma, Jiaming and Cui, Zhiqing and Wang, Binwu and Wang, Pengkun and Zhou, Zhengyang and Zhao, Zhe and Wang, Yang},
  journal={International Joint Conference on Artificial Intelligence},
  year={2025}
}
```
