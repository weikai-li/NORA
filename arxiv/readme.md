## About

This directory contains codes for training GNN models on the ogbn-arxiv dataset.

- GNN models included in this directory: GCN, GraphSAGE, GAT, GCNII, DrGAT (only node classification)
- Datasets included in this directory: ogbn-arxiv

Except for the DrGAT model, the other four GNN models are based on the implementations of the dgl library [1], and their codes are in the "./others" directory. The DrGAT model is based on its GitHub repo (https://github.com/anonymousaabc/DRGCN), and its codes are in the "./DrGAT" directory.

[1] Wang, Minjie, et al. "Deep graph library: A graph-centric, highly-performant package for graph neural networks." *arXiv preprint arXiv:1909.01315* (2019).



## Train the DrGAT model

First, enter the "./DrGAT" directory. We only support training DrGAT for node classification. We have provided the scripts in "./DrGAT/run.sh". Please use the script to first train the teacher model and then the student model. Change "--cycle" to 0, 1, 2, 3, 4 to manually cycle the data split and train the model for five times. At the first time of running the codes, the dataset is automatically downloaded. After training, the trained GNNs are saved in the "./saved_model" directory.



## Train other GNN models

First, enter the "./others" directory. Training a node classification model:

```shell
python train.py --model GCN
```

Training a link prediction model:

```shell
python train.py --model GCN --link_prediction
```

You can substitute the model name with your desired one. Our codes automatically load the tuned hyper-parameters from "./best_config.py". If you want to set a different hyper-parameter, please set them in the args like "--hidden_size".

At the first time of running the codes, the dataset is automatically downloaded. After training, the trained GNNs are saved in the "./saved_model" directory.
