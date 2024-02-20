python main.py --dataset ogbn-arxiv --method nora --model GCN --self_buff 8 --grad_norm 1 \
    --k1 1 --k2 0.8 0 --k3 2
python evaluate.py --dataset ogbn-arxiv --model GCN --val_ratio 0.2
python main.py --dataset ogbn-arxiv --method nora --model GraphSAGE --self_buff 5 --grad_norm 1 \
    --k1 0.8 --k2 0.8 0 --k3 1
python evaluate.py --dataset ogbn-arxiv --model GraphSAGE --val_ratio 0.2
python main.py --dataset ogbn-arxiv --method nora --model DrGAT --self_buff 15 --grad_norm 2 \
    --k1 0.5 --k2 0.8 0 --k3 100
python evaluate.py --dataset ogbn-arxiv --model DrGAT --val_ratio 0.2
python main.py --dataset P50 --method nora --model TIMME --self_buff 20 --grad_norm 5 \
    --k1 0 --k2 0 0 --k3 0.3
python evaluate.py --dataset P50 --model TIMME --val_ratio 0.2
python main.py --dataset P_20_50 --method nora --model TIMME --self_buff 25 --grad_norm 5 \
    --k1 0 --k2 0 0 --k3 0.7
python evaluate.py --dataset P_20_50 --model TIMME --val_ratio 0.2
