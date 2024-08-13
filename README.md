# NORA

Thank you for viewing this repository! This is the official repository of the WWW'24 paper "Fast Inference of Removal-Based Node Influence". We evaluate the task-specific node influence on GNN model’s prediction based on node removal. We use graph neural network (GNN) models as a surrogate to learn the underlying message propagation patterns on a graph. After training a GNN model, we remove a node, apply the trained GNN model on the modified graph, and use the output change to measure the influence of the removed node. NORA (**NO**de-**R**emoval-based f**A**st GNN inference) is an efficient calculation method that can approximate the node influence for all nodes based on gradient and heuristics. Our implementation is based on the [dgl](https://www.dgl.ai/) library [1], an implementation [GitHub repository](https://github.com/anonymousaabc/DRGCN) of DrGAT, and the official [GitHub repository](https://github.com/PatriciaXiao/TIMME) of TIMME [2].

[1] Wang, Minjie, et al. "Deep graph library: A graph-centric, highly-performant package for graph neural networks." arXiv preprint arXiv:1909.01315 (2019).

[2] Zhiping Xiao, Weiping Song, Haoyan Xu, Zhicheng Ren, and Yizhou Sun. 2020. TIMME: Twitter ideology-detection via multi-task multi-relational embedding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2258–2268.



## Train the GNN models

- For the Planetoid datasets (Cora, CiteSeer, and PubMed), please refer to the readme of the "./planetoid" directory.
- For the ogbn-arxiv dataset, please refer to the readme of the "./arxiv" directory.
- For the Twitter datasets (P50, P_20_50), please refer to the readme of the "./TIMME" directory.



## Calculate the node influence

##### Ground truth

To generate the ground truth node influence by the brute-force method, please run:

```shell
python main.py --dataset Cora --model GCN
```

You can substitute the dataset name and model name with your desired one. If you want to use the link prediction model, please add "_edge" after the model number, such as "GCN_edge", "GraphSAGE_edge", etc. Here we provide a list of supported dataset names and GNN model names:

- "Cora", "CiteSeer", and "PubMed" datasets: "GCN", "GraphSAGE", "GAT", "GCNII", "GCN_edge", "GraphSAGE_edge", "GAT_edge", "GCNII_edge"
- "ogbn-arxiv" dataset: "GCN", "GraphSAGE", "DrGAT", "GCNII", "GCN_edge", "GraphSAGE_edge", "GAT_edge", "GCNII_edge"
- "P50" and "P_20_50" datasets: "TIMME", "TIMME_edge"

##### NORA

We provide the script with our hyper-parameters in "nora.sh". Please choose the one according to the dataset and GNN model you want to use.

##### Node mask baseline method

We provide the script with our hyper-parameters in "mask.sh". Please choose the one according to the dataset and GNN model you want to use.

##### Prediction baseline method

We provide the script with our hyper-parameters in "gcn_n.sh" and "gcn_e.sh". "gcn_n.sh" is the "Predict-N" method, and "gcn_e.sh" is the "Predict-E" method. Please choose the one according to the dataset and GNN model you want to use.

##### Change the hyper-parameters

If you want to experiment with other hyper-parameters, please refer to the annotations in the "args" settings in main.py.

##### Evaluate the results

We use evaluate.py for evaluation. If you want to evaluate the approximation performance, we provide the evaluation script in "nora.sh", "mask.sh", "gcn_n.sh", and "gcn_e.sh". Please choose the one according to the dataset and GNN model you want to use. For other evaluation functions, please see the "args" settings in evaluate.py.



## Citation

Thank you for being interested in our paper!

```
@inproceedings{li2024fast,
  title={Fast Inference of Removal-Based Node Influence},
  author={Li, Weikai and Xiao, Zhiping and Luo, Xiao and Sun, Yizhou},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={422--433},
  year={2024}
}
```
