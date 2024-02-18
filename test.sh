python main.py --method gcn-e --model GCN --dataset ogbn-arxiv --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE --dataset ogbn-arxiv --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model DrGAT --dataset ogbn-arxiv --lr 5e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model DrGAT --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model TIMME --dataset P50 --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model TIMME --dataset P50 --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model TIMME --dataset P_20_50 --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model TIMME --dataset P_20_50 --train_ratio 0.15 --val_ratio 0.05
