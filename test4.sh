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

