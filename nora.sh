# Node classification - GCN model / TIMME model (on Twitter datasets)
python main.py --dataset Cora --method gradient --model GCN --k1 0.9 --k2 0.7 --k3 100 --decay 0.95 --self_buff 8
python evaluate.py --dataset Cora --model GCN
python main.py --dataset CiteSeer --method gradient --model GCN --k1 0.8 --k2 1 --k3 10 --decay 0.95 --self_buff 7
python evaluate.py --dataset CiteSeer --model GCN
python main.py --dataset PubMed --method gradient --model GCN --k1 0.4 --k2 1.0 --k3 500 --decay 0.95 --self_buff 30
python evaluate.py --dataset PubMed --model GCN
python main.py --dataset ogbn-arxiv --method gradient --model GCN --n_layers 3 --k1 1.0 --k2 1.0 --k3 1.5e4 --decay 0.8 --self_buff 8
python evaluate.py --dataset ogbn-arxiv --model GCN
python main.py --method gradient --model TIMME --dataset P50 --k1 0.03 --k2 0.07 --k3 6e3 --decay 1.0 --self_buff 3
python evaluate.py --model TIMME --dataset P50
python main.py --method gradient --model TIMME --dataset P_20_50 --k1 0.0 --k2 0.1 --k3 5e3 --decay 0.95 --self_buff 5
python evaluate.py --model TIMME --dataset P_20_50

# Link prediction - GCN model / TIMME model (on Twitter datasets)
python main.py --dataset Cora --method gradient --model GCN-edge --k1 0.9 --k2 0.7 --k3 100 --decay 0.95 --self_buff 8
python evaluate.py --dataset Cora --model GCN-edge
python main.py --dataset CiteSeer --method gradient --model GCN-edge --k1 0.8 --k2 1 --k3 10 --decay 0.95 --self_buff 7
python evaluate.py --dataset CiteSeer --model GCN-edge
python main.py --dataset PubMed --method gradient --model GCN-edge --k1 0.4 --k2 1.0 --k3 500 --decay 0.95 --self_buff 30
python evaluate.py --dataset PubMed --model GCN-edge
python main.py --dataset ogbn-arxiv --method gradient --model GCN-edge --k1 1.0 --k2 1.0 --k3 1.5e4 --decay 0.8 --self_buff 8
python evaluate.py --dataset ogbn-arxiv --model GCN-edge
python main.py --method gradient --model TIMME-edge --dataset P50 --k1 0.03 --k2 0.07 --k3 6e3 --decay 1.0 --self_buff 3
python evaluate.py --model TIMME-edge --dataset P50
python main.py --method gradient --model TIMME-edge --dataset P_20_50 --k1 0.0 --k2 0.1 --k3 5e3 --decay 0.95 --self_buff 5
python evaluate.py --model TIMME-edge --dataset P_20_50

# Node classification - Other models on Planetoid datasets
python main.py --dataset Cora --method gradient --model GraphSAGE --k1 0.7 --k2 0.6 --k3 200 --decay 1 --self_buff 6 --backward_choice out
python evaluate.py --dataset Cora --model GraphSAGE
python main.py --dataset Cora --method gradient --model GAT --k1 0.6 --k2 0.6 --k3 200 --decay 0.9 --self_buff 2
python evaluate.py --dataset Cora --model GAT
python main.py --dataset CiteSeer --method gradient --model GraphSAGE --k1 1.0 --k2 1.0 --k3 400 --decay 0.95 --self_buff 6 --backward_choice out
python evaluate.py --dataset CiteSeer --model GraphSAGE
python main.py --dataset CiteSeer --method gradient --model GAT --k1 0.9 --k2 0.9 --k3 5 --decay 0.9 --self_buff 5 --backward_choice out
python evaluate.py --dataset CiteSeer --model GAT
python main.py --dataset PubMed --method gradient --model GraphSAGE --k1 0.2 --k2 1.0 --k3 5000 --decay 0.95 --self_buff 40 --backward_choice out
python evaluate.py --dataset PubMed --model GraphSAGE
python main.py --dataset PubMed --method gradient --model GAT --k1 0.5 --k2 0.4 --k3 120 --decay 0.95 --self_buff 7 --backward_choice out
python evaluate.py --dataset PubMed --model GAT

# Node classification - Other models on ogbn-arxiv
python main.py --dataset ogbn-arxiv --method gradient --model GraphSAGE --k1 1.0 --k2 1.0 --k3 2e4 --decay 0.95 --self_buff 6 --backward_choice out
python evaluate.py --dataset ogbn-arxiv --model GraphSAGE
python main.py --dataset ogbn-arxiv --method gradient --model DrGAT --k1 1.0 --k2 1.0 --k3 1.5e4 --decay 0.95 --self_buff 2 --backward_choice out
python evaluate.py --dataset ogbn-arxiv --model DrGAT
