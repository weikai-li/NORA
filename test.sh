python main.py --dataset ogbn-arxiv --method mask --model GAT_edge --alpha 500 --lr 1e-2 --wd 0 --num_epochs 50 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GAT_edge --method mask --val_ratio 0.1
