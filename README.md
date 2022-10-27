# GREGNETS

Shibal Ibrahim, Wenyu Chen, Yada Zhu, Pin-Yu Chen, Yang Zhang and Rahul Mazumder

## Introduction

GREGNETS (Graph Regularized Networked Time Series) is an end-to-end learning framework for simultaneous learning of multi-variate forecasting and correlation structure estimation. Joint training leads to computational challenges. We address this by developing a pseudo-likelihood layer, which can be easily appended to any existing multivariate time-series forecasting architecture trainable with (stochastic) gradient descent. The toolkit supports various type of forecasting models (VAR, GCNs, N-GCNs, T-GCNs, LSTMs) and various regularization schemes for the error correlation structure (Lasso, Adaptive-Lasso, Knowledge-Graph based regularizers). 

See our paper [Knowledge Graph Guided Simultaneous Forecasting and Network Learning for Multivariate Financial Time Series](https://dl.acm.org/doi/10.1145/3533271.3561702) appearing in ACM International Conference on AI in Finance 2022 for details.

## Installation
GREGNETS is written in Tensorflow 2.4. Before installing GREGNETS, please make sure that Tensorflow-GPU 2 is installed.

## Support
The toolkit supports the following models:
1. VAR-PC
2. GCN-PC
3. N-GCN-PC
4. T-GCN-PC
5. LSTM-PC

Additionally, the codebase supports KG-masked regularizers with hard/soft masking for partial correlation estimation inferred from KG similarity matrix.

We consider SP500 and SP1500 daily volatilities multivariate time-series.

## To reproduce the VAR-PC numbers in the paper, the tuning scripts src/VARPC/VARPC_tuning.py can be run as follows:
```
python src/VARPC/VARPC_tuning.py --load_directory /home/gridsan/shibal/FinancialForecasting_shared/data/ --cohort 'SP1500' --time_series 'volatilities' \
--regularizer 'Lasso' --KG_mask 'soft' --mask_sparsity 200  \
--num_training_years 2 \
--n_steps 1 \
--version 2 
```

## To reproduce the N-GCN-PC numbers in the paper, the tuning scripts src/NNPC/NGCNPC-tuning.py can be run as follows:
```
python src/NNPC/NGCNPC-Tuning.py \
--load_directory '/home/gridsan/shibal/FinancialForecasting_shared/data/' --time_series 'volatilities' --cohort 'SP1500' \
--num_training_years 2 \
--no-mask \
--ntrials 500 \
--version 1
```

## Citing GREGNETS
If you find this work useful in your research, please consider citing the following paper:

```
@inproceedings{Ibrahim2022,
    author = {Ibrahim, Shibal and Chen, Wenyu and Zhu, Yada and Chen, Pin-Yu and Zhang, Yang and Mazumder, Rahul},
    title = {Knowledge Graph Guided Simultaneous Forecasting and Network Learning for Multivariate Financial Time Series},
    year = {2022},
    isbn = {9781450393768},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3533271.3561702},
    doi = {10.1145/3533271.3561702},
    booktitle = {3rd ACM International Conference on AI in Finance},
    pages = {480–488},
    numpages = {9},
    keywords = {multivariate time-series, sparsity, financial markets, graph neural networks, knowledge graphs, precision matrix},
    location = {New York, NY, USA},
    series = {ICAIF '22}
}
```

