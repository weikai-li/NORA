python main.py --dataset ogbn-arxiv --method nora --model DrGAT --self_buff 15 --grad_norm 2 \
    --k1 0.5 --k2 0.8 0 --k3 0
python evaluate.py --dataset ogbn-arxiv --model DrGAT --val_ratio 0.2
python main.py --dataset ogbn-arxiv --method nora --model DrGAT --self_buff 15 --grad_norm 2 \
    --k1 0.5 --k2 0.8 0 --k3 0
python evaluate.py --dataset ogbn-arxiv --model DrGAT --val_ratio 0.2
# python main.py --dataset ogbn-arxiv --method nora --model DrGAT --self_buff 20 --grad_norm 2 \
    # --k1 0.5 --k2 0.8 0 --k3 0
# python evaluate.py --dataset ogbn-arxiv --model DrGAT --val_ratio 0.2
