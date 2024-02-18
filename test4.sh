python main.py --method gcn-e --model TIMME_edge --dataset P_20_50 --lr 3e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model TIMME_edge --dataset P_20_50 --train_ratio 0.15 --val_ratio 0.05
