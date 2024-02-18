python main.py --method gcn-e --model GCN_edge --dataset ogbn-arxiv --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN_edge --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
