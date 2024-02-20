# Node classification: Planetoid datasets
python main.py --dataset Cora --method nora --model GCN --self_buff 8 --grad_norm 1 \
    --k1 1.0 --k2 0.5 0.0 --k3 1
python evaluate.py --dataset Cora --model GCN --val_ratio 0.2
python main.py --dataset Cora --method nora --model GraphSAGE --self_buff 20 --grad_norm 2 \
    --k1 1.0 --k2 0.0 0.5 --k3 1.5
python evaluate.py --dataset Cora --model GraphSAGE --val_ratio 0.2
python main.py --dataset Cora --method nora --model GAT --self_buff 10 --grad_norm 1 \
    --k1 1.0 --k2 0.0 0.5 --k3 1.5
python evaluate.py --dataset Cora --model GAT --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GCN --self_buff 10 --grad_norm 1 \
    --k1 1.0 --k2 0.8 0.2 --k3 2
python evaluate.py --dataset CiteSeer --model GCN --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GraphSAGE --self_buff 15 --grad_norm 1 \
    --k1 1.0 --k2 0 0.8 --k3 3
python evaluate.py --dataset CiteSeer --model GraphSAGE --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GAT --self_buff 15 --grad_norm 1 \
    --k1 1 --k2 0.8 0 --k3 4
python evaluate.py --dataset CiteSeer --model GAT --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GCN --self_buff 10 --grad_norm 1 \
    --k1 0.4 --k2 0.5 0.0 --k3 1
python evaluate.py --dataset PubMed --model GCN --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GraphSAGE --self_buff 10 --grad_norm 1 \
    --k1 0.2 --k2 0.9 0 --k3 1
python evaluate.py --dataset PubMed --model GraphSAGE --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GAT --self_buff 0 --grad_norm 1 \
    --k1 1 --k2 0.2 0 --k3 1
python evaluate.py --dataset PubMed --model GAT --val_ratio 0.2

# Node classification: ogbn-arxiv and Twitter datasets
python main.py --dataset ogbn-arxiv --method nora --model GCN --self_buff 8 --grad_norm 1 \
    --k1 1 --k2 0.8 0 --k3 2
python evaluate.py --dataset ogbn-arxiv --model GCN --val_ratio 0.2
python main.py --dataset ogbn-arxiv --method nora --model GraphSAGE --self_buff 5 --grad_norm 1 \
    --k1 0.8 --k2 0.8 0 --k3 1
python evaluate.py --dataset ogbn-arxiv --model GraphSAGE --val_ratio 0.2
python main.py --dataset ogbn-arxiv --method nora --model DrGAT --self_buff 15 --grad_norm 2 \
    --k1 0.5 --k2 0.8 0 --k3 0
python evaluate.py --dataset ogbn-arxiv --model DrGAT --val_ratio 0.2
python main.py --dataset P50 --method nora --model TIMME --self_buff 20 --grad_norm 5 \
    --k1 0 --k2 0 0 --k3 0.3
python evaluate.py --dataset P50 --model TIMME --val_ratio 0.2
python main.py --dataset P_20_50 --method nora --model TIMME --self_buff 25 --grad_norm 5 \
    --k1 0 --k2 0 0 --k3 0.7
python evaluate.py --dataset P_20_50 --model TIMME --val_ratio 0.2


# Link prediction: Planetoid datasets
python main.py --dataset Cora --method nora --model GCN_edge --self_buff 2 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 1
python evaluate.py --dataset Cora --model GCN_edge --val_ratio 0.2
python main.py --dataset Cora --method nora --model GraphSAGE_edge --self_buff 0 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 3
python evaluate.py --dataset Cora --model GraphSAGE_edge --val_ratio 0.2
python main.py --dataset Cora --method nora --model GAT_edge --self_buff 0 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 1
python evaluate.py --dataset Cora --model GAT_edge --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GCN_edge --self_buff 20 --grad_norm 2 \
    --k1 0.8 --k2 0 0 --k3 5
python evaluate.py --dataset CiteSeer --model GCN_edge --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GraphSAGE_edge --self_buff 0.1 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 3
python evaluate.py --dataset CiteSeer --model GraphSAGE_edge --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GAT_edge --self_buff 0.1 --k3 5 \
    --k1 1 --k2 0 0
python evaluate.py --dataset CiteSeer --model GAT_edge --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GCN_edge --self_buff 2 --grad_norm 1 \
    --k1 1 --k2 0.8 0 --k3 3
python evaluate.py --dataset PubMed --model GCN_edge --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GraphSAGE_edge --self_buff 0 --grad_norm 2 \
    --k1 1 --k2 0.5 0 --k3 3
python evaluate.py --dataset PubMed --model GraphSAGE_edge --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GAT_edge --self_buff 3 --grad_norm 2 \
    --k1 1 --k2 0.8 0 --k3 1.5
python evaluate.py --dataset PubMed --model GAT_edge --val_ratio 0.2

# Link prediction: ogbn-arxiv and Twitter datasets
python main.py --dataset ogbn-arxiv --method nora --model GCN_edge --self_buff 10 --grad_norm 1 \
    --k1 1 --k2 0.6 0 --k3 5
python evaluate.py --dataset ogbn-arxiv --model GCN_edge --val_ratio 0.2
python main.py --dataset ogbn-arxiv --method nora --model GraphSAGE_edge --self_buff 0 --grad_norm 1 \
    --k1 1 --k2 0 0 --k3 10
python evaluate.py --dataset ogbn-arxiv --model GraphSAGE_edge --val_ratio 0.2
python main.py --dataset ogbn-arxiv --method nora --model GAT_edge --self_buff 0 --grad_norm 1 \
    --k1 1 --k2 0 0 --k3 3
python evaluate.py --dataset ogbn-arxiv --model GAT_edge --val_ratio 0.2
python main.py --dataset P50 --method nora --model TIMME_edge --self_buff 2 --grad_norm 1 \
    --k1 0.8 --k2 1 0 --k3 0.05
python evaluate.py --dataset P50 --model TIMME_edge --val_ratio 0.2
python main.py --dataset P_20_50 --method nora --model TIMME_edge --self_buff 0.5 --grad_norm 5 \
    --k1 1 --k2 1 0 --k3 0.3
python evaluate.py --dataset P_20_50 --model TIMME_edge --val_ratio 0.2
