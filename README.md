<div align="center">

# CauAir: Causal Learning Meet Covariates for Nationwide Air Quality Forecasting

[![Venue IJCAI%202025](https://img.shields.io/badge/Venue-IJCAI%202025-1f6feb)](https://www.ijcai.org/proceedings/2025/353)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c.svg)](https://pytorch.org/)

<p align="center">
  <a href="https://www.ijcai.org/proceedings/2025/353">Paper</a> |
  <a href="#largeaq-download">LargeAQ</a> |
  <a href="#running-experiments">Running</a> |
  <a href="#-citation">Citation</a> |
  <a href="#-license">License</a>
</p>

</div>

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

> [!NOTE]
> **LargeAQ** is released under [CC BY 4.0](LICENSE-DATA): free for academic and commercial use, with attribution. The KnowAir and CCAQ data below are third-party and keep their original terms.

### LargeAQ

LargeAQ is the nationwide, long-term air-quality dataset introduced for CauAir: hourly AQI (including PM2.5, PM10, NO2) from 1,341 monitoring stations in 333 Chinese cities, January 2016 to December 2023, plus five weather covariates (temperature, humidity, precipitation, wind shear, wind speed).

**Data sources.** The air-quality records come from the [China National Environmental Monitoring Centre (CNEMC)](https://www.cnemc.cn/), and the weather covariates were collected from the Chinese Weather Website, as described in the paper. The CC BY 4.0 license covers our compilation, cleaning and organization of the dataset; it does not claim ownership of the original upstream measurements.

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

## 📄 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{ma2025causal,
  title     = {Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting},
  author    = {Ma, Jiaming and Cui, Zhiqing and Wang, Binwu and Wang, Pengkun and Zhou, Zhengyang and Zhao, Zhe and Wang, Yang},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {3171--3179},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/353},
  url       = {https://www.ijcai.org/proceedings/2025/353}
}
```

## 🙏 Acknowledgements

Includes processed [KnowAir](https://github.com/shuowang-ai/PM2.5-GNN) and [CCAQ / GAGNN](https://github.com/Friger/GAGNN) data for comparison; both keep the license of their original release.

## 📜 License

| Component | License |
| --- | --- |
| Code | [MIT License](LICENSE) |
| **LargeAQ dataset** | [Creative Commons Attribution 4.0 (CC BY 4.0)](LICENSE-DATA): free to share and adapt, including for commercial use, provided you give credit by citing the paper above. Upstream sources: CNEMC and the Chinese Weather Website |
| KnowAir, CCAQ (included for comparison) | Original terms of their upstream releases |
