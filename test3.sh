python main.py --dataset ogbn-arxiv --method nora --model GCNII_edge --self_buff 10 --grad_norm 1 \
    --k1 0.6 --k2 0 0 --k3 10000
python evaluate.py --dataset ogbn-arxiv --model GCNII_edge --val_ratio 0.1