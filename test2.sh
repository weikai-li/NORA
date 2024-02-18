python main.py --method nora --model TIMME --dataset P50 --k1 0.03 --k2 0.07 --k3 6e3 --decay 1.0 --self_buff 3
python evaluate.py --model TIMME --dataset P50 --val_ratio 0.2
python main.py --method nora --model TIMME --dataset P_20_50 --k1 0.0 --k2 0.1 --k3 5e3 --decay 0.95 --self_buff 5
python evaluate.py --model TIMME --dataset P_20_50 --val_ratio 0.2
