python main.py --dataset ogbn-arxiv --method nora --model GCNII --self_buff 20 --grad_norm 1 \
    --k1 0.6 --k2 0.6 0 --k3 10000
python evaluate.py --dataset ogbn-arxiv --model GCNII --val_ratio 0.1