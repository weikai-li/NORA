python main.py --dataset Cora --method nora --model GCN_edge --k1 0.9 --k2 0.7 --k3 100 --decay 0.95 --self_buff 8
python evaluate.py --dataset Cora --model GCN_edge --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GCN_edge --k1 0.8 --k2 1 --k3 10 --decay 0.95 --self_buff 7
python evaluate.py --dataset CiteSeer --model GCN_edge --val_ratio 0.2
python main.py --method nora --model TIMME_edge --dataset P50 --k1 0.03 --k2 0.07 --k3 6e3 --decay 1.0 --self_buff 3
python evaluate.py --model TIMME_edge --dataset P50 --val_ratio 0.2
python main.py --method nora --model TIMME_edge --dataset P_20_50 --k1 0.0 --k2 0.1 --k3 5e3 --decay 0.95 --self_buff 5
python evaluate.py --model TIMME_edge --dataset P_20_50 --val_ratio 0.2
