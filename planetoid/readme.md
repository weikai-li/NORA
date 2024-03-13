## About

This directory contains codes for training GNN models on the Planetoid datasets. The GNN models are based on the implementations of the [dgl](https://www.dgl.ai/) library [1].

- GNN models included in this directory: GCN, GraphSAGE, GAT, GCNII
- Datasets included in this directory: Cora, CiteSeer, PubMed

[1] Wang, Minjie, et al. "Deep graph library: A graph-centric, highly-performant package for graph neural networks." arXiv preprint arXiv:1909.01315 (2019).



## Train the GNN models

Training a node classification model:

```shell
python train.py --dataset Cora --model GCN
```

Training a link prediction model:

```shell
python train.py --dataset Cora --model GCN --link_prediction
```

You can substitute the dataset name and model name with your desired one. Our codes automatically load the tuned hyper-parameters from "./best_config.py". If you want to set a different hyper-parameter, please set them in the args like "--hidden_size".

At the first time of running the codes, the dataset is automatically downloaded. After training, the trained GNNs are saved in the "./saved_model" directory.
