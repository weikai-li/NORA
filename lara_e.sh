# Node classification - GCN model / TIMME model (on Twitter datasets)
python main.py --method lara-e-gcn --model GCN --dataset Cora --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN --dataset Cora
python main.py --method lara-e-gcn --model GCN --dataset CiteSeer --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN --dataset CiteSeer
python main.py --method lara-e-gcn --model GCN --dataset PubMed --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN --dataset PubMed
python main.py --method lara-e-gcn --model GCN --dataset ogbn-arxiv --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN --dataset ogbn-arxiv
python main.py --method lara-e-gcn --model TIMME --dataset P50 --lr 1e-3 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model TIMME --dataset P50
python main.py --method lara-e-gcn --model TIMME --dataset P_20_50 --lr 1e-3 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model TIMME --dataset P_20_50

# Link prediction - GCN model / TIMME model (on Twitter datasets)
python main.py --method lara-e-gcn --model GCN-edge --dataset Cora --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN-edge --dataset Cora
python main.py --method lara-e-gcn --model GCN-edge --dataset CiteSeer --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN-edge --dataset CiteSeer
python main.py --method lara-e-gcn --model GCN-edge --dataset PubMed --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN-edge --dataset PubMed
python main.py --method lara-e-gcn --model GCN-edge --dataset ogbn-arxiv --lr 1e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GCN-edge --dataset ogbn-arxiv
python main.py --method lara-e-gcn --model TIMME-edge --dataset P50 --lr 1e-3 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model TIMME-edge --dataset P50
python main.py --method lara-e-gcn --model TIMME-edge --dataset P_20_50 --lr 1e-3 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model TIMME-edge --dataset P_20_50

# Node classification - Other models on Planetoid datasets
python main.py --method lara-e-gcn --model GraphSAGE --dataset Cora --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GraphSAGE --dataset Cora
python main.py --method lara-e-gcn --model GAT --dataset Cora --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GAT --dataset Cora
python main.py --method lara-e-gcn --model GraphSAGE --dataset CiteSeer --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GraphSAGE --dataset CiteSeer
python main.py --method lara-e-gcn --model GAT --dataset CiteSeer --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GAT --dataset CiteSeer

# Node classification - Other models on ogbn-arxiv
python main.py --method lara-e-gcn --model GraphSAGE --dataset ogbn-arxiv --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model GraphSAGE --dataset ogbn-arxiv
python main.py --method lara-e-gcn --model DrGAT --dataset ogbn-arxiv --lr 5e-4 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-e-gcn --model DrGAT --dataset ogbn-arxiv
