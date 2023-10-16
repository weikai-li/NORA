# Node classification - GCN model / TIMME model (on Twitter datasets)
python main.py --dataset Cora --method mask --model GCN --alpha 0.05 --lr 5e-4 --n_epochs 100
python evaluate.py --dataset Cora --model GCN --method mask
python main.py --dataset CiteSeer --method mask --model GCN --alpha 0.2 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset CiteSeer --model GCN --method mask
python main.py --dataset PubMed --method mask --model GCN --alpha 0.2 --lr 2e-3 --n_epochs 100
python evaluate.py --dataset PubMed --model GCN --method mask
python main.py --dataset ogbn-arxiv --method mask --model GCN --alpha 2 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset ogbn-arxiv --model GCN --method mask
python main.py --dataset P50 --method mask --model TIMME --alpha 10 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset P50 --model TIMME --method mask
python main.py --dataset P_20_50 --method mask --model TIMME --alpha 1 --lr 2e-3 --n_epochs 100
python evaluate.py --dataset P_20_50 --model TIMME --method mask

# Link prediction - GCN model / TIMME model (on Twitter datasets)
python main.py --dataset Cora --method mask --model GCN-edge --alpha 0.001 --lr 1e-4 --n_epochs 100
python evaluate.py --dataset Cora --model GCN-edge --method mask
python main.py --dataset CiteSeer --method mask --model GCN-edge --alpha 0.01 --lr 1e-4 --n_epochs 100
python evaluate.py --dataset CiteSeer --model GCN-edge --method mask
python main.py --dataset PubMed --method mask --model GCN-edge --alpha 0.2 --lr 2e-3 --n_epochs 100
python evaluate.py --dataset PubMed --model GCN-edge --method mask
python main.py --dataset ogbn-arxiv --method mask --model GCN-edge --alpha 2 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset ogbn-arxiv --model GCN-edge --method mask
python main.py --dataset P50 --method mask --model TIMME-edge --alpha 0.2 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset P50 --model TIMME-edge --method mask
python main.py --dataset P_20_50 --method mask --model TIMME-edge --alpha 2 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset P_20_50 --model TIMME-edge --method mask

# Node classification - Other models on Planetoid datasets
python main.py --dataset Cora --method mask --model GraphSAGE --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset Cora --model GraphSAGE --method mask
python main.py --dataset Cora --method mask --model GAT --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset Cora --model GAT --method mask
python main.py --dataset CiteSeer --method mask --model GraphSAGE --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset CiteSeer --model GraphSAGE --method mask
python main.py --dataset CiteSeer --method mask --model GAT --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset CiteSeer --model GAT --method mask
python main.py --dataset PubMed --method mask --model GraphSAGE --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset PubMed --model GraphSAGE --method mask
python main.py --dataset PubMed --method mask --model GAT --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset PubMed --model GAT --method mask

# Node classification - Other models on ogbn-arxiv
python main.py --dataset ogbn-arxiv --method mask --model GraphSAGE --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset ogbn-arxiv --model GraphSAGE --method mask
python main.py --dataset ogbn-arxiv --method mask --model DrGAT --alpha 0.5 --lr 1e-3 --n_epochs 100
python evaluate.py --dataset ogbn-arxiv --model DrGAT --method mask
