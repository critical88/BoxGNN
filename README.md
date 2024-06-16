# BoxGNN

This is our PyTorch implementation for the paper:

> Fake Lin, Ziwei Zhao, Xi Zhu, Da Zhang, Shitian Shen, Xueying Li, Tong Xu, Suojuan Zhang and Enhong Chen (2024). When Box Meets Graph Neural Network in Tag-aware Recommendation. In KDD ’24, August 25–29, 2024, Barcelona, Spain.

### Reproducibility
MovieLens
```shell
python main.py --dataset movielens --model boxgnn --beta 0.2
```
LastFm
```shell
python main.py --dataset lastfm --model boxgnn --beta 0.3
```

E-shop
```shell
python main.py --dataset e-shop --model boxgnn --beta 0.2
```

### Preprocess

The preprocess details are listed in `utils/preprocess.py`, and the raw data of movielens and lastfm come from [hetrec-2011](https://grouplens.org/datasets/hetrec-2011). Notably, we randomly select 80%, 10%, 10% of data as training set, validation set and test set, respectively. We only use data in training set to construct collaborative tag graph.