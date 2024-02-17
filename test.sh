python main.py --dataset ogbn-arxiv --method nora --model GCN-edge --k1 1.0 --k2 1.0 --k3 1.5e4 --decay 0.8 --self_buff 8
python evaluate.py --dataset ogbn-arxiv --model GCN-edge
