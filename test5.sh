python main.py --method gcn-n --model GAT_edge --dataset ogbn-arxiv --lr 1e-4 --wd 1e-5 --num_epochs 200 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT_edge --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
