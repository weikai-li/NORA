# Node classification: Planetoid datasets
python main.py --method gcn-e --model GCN --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE --dataset Cora --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GAT --dataset Cora --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GAT --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GCN --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GAT --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GAT --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GCN --dataset PubMed --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE --dataset PubMed --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GAT --dataset PubMed --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GAT --dataset PubMed --train_ratio 0.15 --val_ratio 0.05

# Node classification: ogbn-arxiv and Twitter datasets
python main.py --method gcn-e --model GCN --dataset ogbn-arxiv --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE --dataset ogbn-arxiv --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model DrGAT --dataset ogbn-arxiv --lr 5e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model DrGAT --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model TIMME --dataset P50 --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model TIMME --dataset P50 --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model TIMME --dataset P_20_50 --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model TIMME --dataset P_20_50 --train_ratio 0.15 --val_ratio 0.05


# Link prediction: planetoid datasets
python main.py --method gcn-e --model GCN_edge --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN_edge --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE_edge --dataset Cora --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE_edge --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GAT_edge --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GAT_edge --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GCN_edge --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN_edge --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE_edge --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE_edge --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GAT_edge --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GAT_edge --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GCN_edge --dataset PubMed --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN_edge --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE_edge --dataset PubMed --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE_edge --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GAT_edge --dataset PubMed --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GAT_edge --dataset PubMed --train_ratio 0.15 --val_ratio 0.05

# Link prediction: ogbn-arxiv and Twitter datasets
python main.py --method gcn-e --model GCN_edge --dataset ogbn-arxiv --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GCN_edge --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model GraphSAGE_edge --dataset ogbn-arxiv --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model GraphSAGE_edge --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05

python main.py --method gcn-e --model TIMME_edge --dataset P50 --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model TIMME_edge --dataset P50 --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-e --model TIMME_edge --dataset P_20_50 --lr 3e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-e --model TIMME_edge --dataset P_20_50 --train_ratio 0.15 --val_ratio 0.05
