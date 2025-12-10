# Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Prediction (IJCAI 2025)
This is the official repository of our IJCAI 2025: CauAir. 

## 1. Introduction about the datasets
### 1.1 LargeAQ (Ours)
In our paper, we create LargeAQ, a nationwide and long-term air quality dataset. **Given the constraints of China's data policies, directly open-sourcing this dataset may carry certain negative implications. Researchers interested in using the LargeAQ dataset are kindly requested to send an email to [JiamingMa@mail.ustc.edu.cn] to apply for access. Please include the following information in your email: your institution’s name, your full name, and the intended purpose of use. We will ensure a timely response and will share the dataset with you.**


### 1.2 KnowAir and CCAQ
We implement extra experiments on two open-sourced nationwide air quality datasets, [KnowAir](https://github.com/shuowang-ai/PM2.5-GNN) and [Chinese Cities Air Quality (CCAQ)](https://github.com/Friger/GAGNN). We have already processed the data from these two open-source datasets, and no additional operations are required to train the model on them.

<br>

## 2. Environmental Requirments
The experiment requires the same environment as [LargeST](https://github.com/liuxu77/LargeST/blob/main).

<br>

## 3. Model Running
To run CauAir on <b>LargeAQ</b>, for example, you may irectly execute the Python file in the terminal:
```
python experiments/cauair/main.py --device cuda:YOUR_CUDA_ID --model_name cauair --dataset 24_24 --iput_dim 8 --tod 24 --dim 128 --head 8 --rank 32
```
To run CauAir on <b>KnowAir</b>, you may irectly execute the Python file in the terminal:
```
python experiments/cauair/main.py --device cuda:YOUR_CUDA_ID --model_name cauair --dataset 24_24_KA --iput_dim 13 --tod 8 --dim 128 --head 2 --rank 10
```
To run CauAir on <b>CCAQ</b>, you may directly execute the Python file in the terminal:
```
python experiments/cauair/main.py --device cuda:YOUR_CUDA_ID --model_name cauair --dataset 24_24_G --iput_dim 10 --tod 24 --dim 128 --head 4 --rank 108
```

## 📄 Citation

If you find this project helpful, please cite us:

```bibtex
@article{ma2025causal,
  title={Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting},
  author={Ma, Jiaming and Cui, Zhiqing and Wang, Binwu and Wang, Pengkun and Zhou, Zhengyang and Zhao, Zhe and Wang, Yang}, 
  journal = {International Joint Conference on Artificial Intelligence},
  year={2025}
}
```
