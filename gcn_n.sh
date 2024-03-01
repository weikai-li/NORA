# Node classification: Planetoid datasets
python main.py --dataset Cora --method gcn-n --model GCN --lr 1e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GCN --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset Cora --method gcn-n --model GraphSAGE --lr 3e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GraphSAGE --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset Cora --method gcn-n --model GAT --lr 5e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.2 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GAT --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset Cora --method gcn-n --model GCNII --lr 2e-4 --wd 0 --num_epochs 50 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.7 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GCNII --train_ratio 0.07 --val_ratio 0.03

python main.py --dataset CiteSeer --method gcn-n --model GCN --lr 5e-5 --wd 0 --num_epochs 50 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.2 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GCN --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset CiteSeer --method gcn-n --model GraphSAGE --lr 5e-5 --wd 0 --num_epochs 50 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.2 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GraphSAGE --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset CiteSeer --method gcn-n --model GAT --lr 5e-5 --wd 0 --num_epochs 50 \
    --pred_hidden 256 --pred_num_layers 2 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GAT --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset CiteSeer --method gcn-n --model GCNII --lr 3e-5 --wd 1e-5 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.2 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GCNII --train_ratio 0.07 --val_ratio 0.03

python main.py --dataset PubMed --method gcn-n --model GCN --lr 1e-2 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset PubMed --method gcn-n --model GCN --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset PubMed --method gcn-n --model GraphSAGE --lr 1e-3 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset PubMed --method gcn-n --model GraphSAGE --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset PubMed --method gcn-n --model GAT --lr 5e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset PubMed --method gcn-n --model GAT --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset PubMed --method gcn-n --model GCNII --lr 5e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset PubMed --method gcn-n --model GCNII --train_ratio 0.07 --val_ratio 0.03

# Node classification: ogbn-arxiv and Twitter datasets
python main.py --dataset ogbn-arxiv --method gcn-n --model GCN --lr 1e-2 --wd 0 --num_epochs 300 \
    --pred_hidden 128 --pred_num_layers 4 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GCN --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset ogbn-arxiv --method gcn-n --model GraphSAGE --lr 1e-2 --wd 0 --num_epochs 300 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GraphSAGE --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset ogbn-arxiv --method gcn-n --model DrGAT --lr 2e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model DrGAT --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset ogbn-arxiv --method gcn-n --model GCNII --lr 1e-2 --wd 0 --num_epochs 400 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GCNII --train_ratio 0.07 --val_ratio 0.03

python main.py --dataset P50 --method gcn-n --model TIMME --lr 5e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset P50 --method gcn-n --model TIMME --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset P_20_50 --method gcn-n --model TIMME --lr 1e-4 --wd 0 --num_epochs 50 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset P_20_50 --method gcn-n --model TIMME --train_ratio 0.07 --val_ratio 0.03


# Link prediction: Planetoid datasets
python main.py --dataset Cora --method gcn-n --model GCN_edge --lr 5e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.2 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GCN_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset Cora --method gcn-n --model GraphSAGE_edge --lr 5e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 64 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GraphSAGE_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset Cora --method gcn-n --model GAT_edge --lr 5e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GAT_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset Cora --method gcn-n --model GCNII_edge --lr 1e-4 --wd 0 --num_epochs 50 \
    --pred_hidden 256 --pred_num_layers 2 --pred_dropout 0.2 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset Cora --method gcn-n --model GCNII_edge --train_ratio 0.07 --val_ratio 0.03

python main.py --dataset CiteSeer --method gcn-n --model GCN_edge --lr 1e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GCN_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset CiteSeer --method gcn-n --model GraphSAGE_edge --lr 5e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.7 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GraphSAGE_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset CiteSeer --method gcn-n --model GAT_edge --lr 5e-5 --wd 0 --num_epochs 50 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GAT_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset CiteSeer --method gcn-n --model GCNII_edge --lr 5e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset CiteSeer --method gcn-n --model GCNII_edge --train_ratio 0.07 --val_ratio 0.03

python main.py --dataset PubMed --method gcn-n --model GCN_edge --lr 1e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset PubMed --method gcn-n --model GCN_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset PubMed --method gcn-n --model GraphSAGE_edge --lr 1e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset PubMed --method gcn-n --model GraphSAGE_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset ogbn-arxiv --method gcn-n --model GAT_edge --lr 5e-5 --wd 5e-5 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GAT_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset PubMed --method gcn-n --model GCNII_edge --lr 1e-4 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset PubMed --method gcn-n --model GCNII_edge --train_ratio 0.07 --val_ratio 0.03

# Link prediction: ogbn-arxiv and Twitter datasets
python main.py --dataset ogbn-arxiv --method gcn-n --model GCN_edge --lr 5e-5 --wd 0 --num_epochs 50 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GCN_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset ogbn-arxiv --method gcn-n --model GraphSAGE_edge --lr 5e-5 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 4 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GraphSAGE_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset ogbn-arxiv --method gcn-n --model GAT_edge --lr 1e-4 --wd 1e-5 --num_epochs 200 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GAT_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset ogbn-arxiv --method gcn-n --model GCNII_edge --lr 5e-5 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset ogbn-arxiv --method gcn-n --model GCNII_edge --train_ratio 0.07 --val_ratio 0.03

python main.py --dataset P50 --method gcn-n --model TIMME_edge --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset P50 --method gcn-n --model TIMME_edge --train_ratio 0.07 --val_ratio 0.03
python main.py --dataset P_20_50 --method gcn-n --model TIMME_edge --lr 5e-3 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.07 --val_ratio 0.03
python evaluate.py --dataset P_20_50 --method gcn-n --model TIMME_edge --train_ratio 0.07 --val_ratio 0.03
