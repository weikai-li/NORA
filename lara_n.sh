# Node classification - GCN model / TIMME model (on Twitter datasets)
python main.py --method lara-n-gcn --model GCN --dataset Cora --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN --dataset Cora
python main.py --method lara-n-gcn --model GCN --dataset CiteSeer --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN --dataset CiteSeer
python main.py --method lara-n-gcn --model GCN --dataset PubMed --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN --dataset PubMed
python main.py --method lara-n-gcn --model GCN --dataset ogbn-arxiv --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN --dataset ogbn-arxiv
python main.py --method lara-n-gcn --model TIMME --dataset P50 --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model TIMME --dataset P50
python main.py --method lara-n-gcn --model TIMME --dataset P_20_50 --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model TIMME --dataset P_20_50

# Link prediction - GCN model / TIMME model (on Twitter datasets)
python main.py --method lara-n-gcn --model GCN-edge --dataset Cora --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN-edge --dataset Cora
python main.py --method lara-n-gcn --model GCN-edge --dataset CiteSeer --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN-edge --dataset CiteSeer
python main.py --method lara-n-gcn --model GCN-edge --dataset PubMed --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN-edge --dataset PubMed
python main.py --method lara-n-gcn --model GCN-edge --dataset ogbn-arxiv --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model GCN-edge --dataset ogbn-arxiv
python main.py --method lara-n-gcn --model TIMME-edge --dataset P50 --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model TIMME-edge --dataset P50
python main.py --method lara-n-gcn --model TIMME-edge --dataset P_20_50 --lr 5e-5 --n_epochs 100 --pred_hidden 64 --pred_n_layers 2 --pred_dropout 0.5
python evaluate.py --method lara-n-gcn --model TIMME-edge --dataset P_20_50
