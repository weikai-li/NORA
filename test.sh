python main.py --dataset Cora --method nora --model GCN --k1 0.9 --k2 0.7 --k3 100 --decay 0.95 --self_buff 8
python evaluate.py --dataset Cora --model GCN --val_ratio 0.2
python main.py --dataset Cora --method nora --model GraphSAGE --k1 0.7 --k2 0.6 --k3 200 --decay 1 --self_buff 6
python evaluate.py --dataset Cora --model GraphSAGE --val_ratio 0.2
python main.py --dataset Cora --method nora --model GAT --k1 0.6 --k2 0.6 --k3 200 --decay 0.9 --self_buff 2
python evaluate.py --dataset Cora --model GAT --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GraphSAGE --k1 0.2 --k2 1.0 --k3 5000 --decay 0.95 --self_buff 40
python evaluate.py --dataset PubMed --model GraphSAGE --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GAT --k1 0.5 --k2 0.4 --k3 120 --decay 0.95 --self_buff 7
python evaluate.py --dataset PubMed --model GAT --val_ratio 0.2
