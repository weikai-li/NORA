# Node classification: Planetoid datasets
python main.py --method gcn-n --model GCN --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE --dataset Cora --lr 5e-3 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GAT --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GCN --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GAT --dataset CiteSeer --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GCN --dataset PubMed --lr 1e-2 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE --dataset PubMed --lr 1e-3 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GAT --dataset PubMed --lr 5e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT --dataset PubMed --train_ratio 0.15 --val_ratio 0.05

# Node classification: ogbn-arxiv and Twitter datasets
python main.py --method gcn-n --model GCN --dataset ogbn-arxiv --lr 5e-4 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 4 --pred_dropout 0.5 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE --dataset ogbn-arxiv --lr 1e-2 --wd 0 --num_epochs 300 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model DrGAT --dataset ogbn-arxiv --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model DrGAT --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model TIMME --dataset P50 --lr 5e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0.5 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model TIMME --dataset P50 --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model TIMME --dataset P_20_50 --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 2 --pred_dropout 0.0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model TIMME --dataset P_20_50 --train_ratio 0.15 --val_ratio 0.05


# Link prediction: Planetoid datasets
python main.py --method gcn-n --model GCN_edge --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN_edge --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE_edge --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE_edge --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GAT_edge --dataset Cora --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT_edge --dataset Cora --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GCN_edge --dataset CiteSeer --lr 3e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN_edge --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE_edge --dataset CiteSeer --lr 5e-4 --wd 0 --num_epochs 200 \
    --pred_hidden 64 --pred_num_layers 2 --pred_dropout 0.7 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE_edge --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GAT_edge --dataset CiteSeer --lr 3e-3 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0.5 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT_edge --dataset CiteSeer --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GCN_edge --dataset PubMed --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN_edge --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE_edge --dataset PubMed --lr 2e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE_edge --dataset PubMed --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GAT_edge --dataset PubMed --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT_edge --dataset PubMed --train_ratio 0.15 --val_ratio 0.05

# Link prediction: ogbn-arxiv and Twitter datasets
python main.py --method gcn-n --model GCN_edge --dataset ogbn-arxiv --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GCN_edge --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GraphSAGE_edge --dataset ogbn-arxiv --lr 1e-3 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GraphSAGE_edge --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model GAT_edge --dataset ogbn-arxiv --lr 1e-4 --wd 1e-5 --num_epochs 200 \
    --pred_hidden 256 --pred_num_layers 3 --pred_dropout 0.3 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model GAT_edge --dataset ogbn-arxiv --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model TIMME_edge --dataset P50 --lr 5e-4 --wd 0 --num_epochs 100 \
    --pred_hidden 256 --pred_num_layers 2 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model TIMME_edge --dataset P50 --train_ratio 0.15 --val_ratio 0.05
python main.py --method gcn-n --model TIMME_edge --dataset P_20_50 --lr 5e-3 --wd 0 --num_epochs 200 \
    --pred_hidden 128 --pred_num_layers 3 --pred_dropout 0 --train_ratio 0.15 --val_ratio 0.05
python evaluate.py --method gcn-n --model TIMME_edge --dataset P_20_50 --train_ratio 0.15 --val_ratio 0.05
