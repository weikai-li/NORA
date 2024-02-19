python main.py --dataset CiteSeer --method nora --model GAT --decay 1.0 --self_buff 5 --k3 0 \
    --k1 1.0 --k2 0.2 0
python evaluate.py --dataset CiteSeer --model GAT --val_ratio 0.2
# python main.py --dataset CiteSeer --method nora --model GAT --decay 1.0 --self_buff 5 --k3 10000 \
#     --k1 1.0 --k2 0.2 0.2
# python evaluate.py --dataset CiteSeer --model GAT --val_ratio 0.2
# python main.py --dataset CiteSeer --method nora --model GAT --decay 1.0 --self_buff 5 --k3 10000 \
#     --k1 1.0 --k2 0.2 0.4
# python evaluate.py --dataset CiteSeer --model GAT --val_ratio 0.2
