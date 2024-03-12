# Node classification: Planetoid datasets
python main.py --dataset Cora --method mask --model GCN --beta 1 --lr 1e-3 --wd 0.5 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset Cora --model GCN --method mask --val_ratio 0.1
python main.py --dataset Cora --method mask --model GraphSAGE --beta 1 --lr 1e-3 --wd 0.1 --num_epochs 300 --val_ratio 0.1
python evaluate.py --dataset Cora --model GraphSAGE --method mask --val_ratio 0.1
python main.py --dataset Cora --method mask --model GAT --beta 200 --lr 1e-3 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset Cora --model GAT --method mask --val_ratio 0.1
python main.py --dataset Cora --method mask --model GCNII --beta 0.1 --lr 1e-3 --wd 1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset Cora --model GCNII --method mask --val_ratio 0.1

python main.py --dataset CiteSeer --method mask --model GCN --beta 10 --lr 2e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GCN --method mask --val_ratio 0.1
python main.py --dataset CiteSeer --method mask --model GraphSAGE --beta 1 --lr 2e-3 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GraphSAGE --method mask --val_ratio 0.1
python main.py --dataset CiteSeer --method mask --model GAT --beta 10 --lr 1e-3 --wd 1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GAT --method mask --val_ratio 0.1
python main.py --dataset CiteSeer --method mask --model GCNII --beta 10 --lr 2e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GCNII --method mask --val_ratio 0.1

python main.py --dataset PubMed --method mask --model GCN --beta 1 --lr 2e-3 --wd 1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GCN --method mask --val_ratio 0.1
python main.py --dataset PubMed --method mask --model GraphSAGE --beta 1 --lr 2e-3 --wd 0.1 --num_epochs 300 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GraphSAGE --method mask --val_ratio 0.1
python main.py --dataset PubMed --method mask --model GAT --beta 10 --lr 2e-3 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GAT --method mask --val_ratio 0.1
python main.py --dataset PubMed --method mask --model GCNII --beta 1 --lr 2e-3 --wd 2 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GCNII --method mask --val_ratio 0.1

# Node classification: ogbn-arxiv and Twitter datasets
python main.py --dataset ogbn-arxiv --method mask --model GCN --beta 1 --lr 2e-3 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GCN --method mask --val_ratio 0.1
python main.py --dataset ogbn-arxiv --method mask --model GraphSAGE --beta 100 --lr 2e-3 --wd 0.1 --num_epochs 300 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GraphSAGE --method mask --val_ratio 0.1
python main.py --dataset ogbn-arxiv --method mask --model DrGAT --beta 500 --lr 2e-3 --wd 0.01 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model DrGAT --method mask --val_ratio 0.1
python main.py --dataset ogbn-arxiv --method mask --model GCNII --beta 1 --lr 1e-2 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GCNII --method mask --val_ratio 0.1

python main.py --dataset P50 --method mask --model TIMME --beta 100 --lr 2e-3 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset P50 --model TIMME --method mask --val_ratio 0.1
python main.py --dataset P_20_50 --method mask --model TIMME --beta 1 --lr 5e-3 --wd 0.5 --num_epochs 300 --val_ratio 0.1
python evaluate.py --dataset P_20_50 --model TIMME --method mask --val_ratio 0.1

# Link prediction: Planetoid datasets
python main.py --dataset Cora --method mask --model GCN_edge --beta 10 --lr 1e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset Cora --model GCN_edge --method mask --val_ratio 0.1
python main.py --dataset Cora --method mask --model GraphSAGE_edge --beta 0 --lr 1e-2 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset Cora --model GraphSAGE_edge --method mask --val_ratio 0.1
python main.py --dataset Cora --method mask --model GAT_edge --beta 100 --lr 5e-4 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset Cora --model GAT_edge --method mask --val_ratio 0.1
python main.py --dataset Cora --method mask --model GCNII_edge --beta 10 --lr 1e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset Cora --model GCNII_edge --method mask --val_ratio 0.1

python main.py --dataset CiteSeer --method mask --model GCN_edge --beta 0 --lr 1e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GCN_edge --method mask --val_ratio 0.1
python main.py --dataset CiteSeer --method mask --model GraphSAGE_edge --beta 0 --lr 1e-3 --wd 4 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GraphSAGE_edge --method mask --val_ratio 0.1
python main.py --dataset CiteSeer --method mask --model GAT_edge --beta 10 --lr 1e-3 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GAT_edge --method mask --val_ratio 0.1
python main.py --dataset CiteSeer --method mask --model GCNII_edge --beta 0.1 --lr 2e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset CiteSeer --model GCNII_edge --method mask --val_ratio 0.1

python main.py --dataset PubMed --method mask --model GCN_edge --beta 1 --lr 1e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GCN_edge --method mask --val_ratio 0.1
python main.py --dataset PubMed --method mask --model GraphSAGE_edge --beta 100 --lr 1e-2 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GraphSAGE_edge --method mask --val_ratio 0.1
python main.py --dataset PubMed --method mask --model GAT_edge --beta 0 --lr 1e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GAT_edge --method mask --val_ratio 0.1
python main.py --dataset PubMed --method mask --model GCNII_edge --beta 1 --lr 2e-3 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset PubMed --model GCNII_edge --method mask --val_ratio 0.1

# Link prediction: ogbn-arxiv and TIMME
python main.py --dataset ogbn-arxiv --method mask --model GCN_edge --beta 0 --lr 1e-2 --wd 0.5 --num_epochs 300 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GCN_edge --method mask --val_ratio 0.1
python main.py --dataset ogbn-arxiv --method mask --model GraphSAGE_edge --beta 0 --lr 1e-3 --wd 20 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GraphSAGE_edge --method mask --val_ratio 0.1
python main.py --dataset ogbn-arxiv --method mask --model GAT_edge --beta 500 --lr 1e-2 --wd 0 --num_epochs 50 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GAT_edge --method mask --val_ratio 0.1
python main.py --dataset ogbn-arxiv --method mask --model GCNII_edge --beta 40 --lr 5e-3 --wd 4e-5 --num_epochs 50 --val_ratio 0.1
python evaluate.py --dataset ogbn-arxiv --model GCNII_edge --method mask --val_ratio 0.1

python main.py --dataset P50 --method mask --model TIMME_edge --beta 0 --lr 5e-4 --wd 0.1 --num_epochs 100 --val_ratio 0.1
python evaluate.py --dataset P50 --model TIMME_edge --method mask --val_ratio 0.1
python main.py --dataset P_20_50 --method mask --model TIMME_edge --beta 0 --lr 1e-3 --wd 0.1 --num_epochs 200 --val_ratio 0.1
python evaluate.py --dataset P_20_50 --model TIMME_edge --method mask --val_ratio 0.1
