python main.py --dataset Cora --method nora --model GCN --self_buff 8 --grad_norm 1 \
    --k1 1.0 --k2 0.5 0.0 --k3 1
python evaluate.py --dataset Cora --model GCN --val_ratio 0.2
python main.py --dataset Cora --method nora --model GraphSAGE --self_buff 20 --grad_norm 2 \
    --k1 1.0 --k2 0.0 0.5 --k3 1.5
python evaluate.py --dataset Cora --model GraphSAGE --val_ratio 0.2
python main.py --dataset Cora --method nora --model GAT --self_buff 10 --grad_norm 1 \
    --k1 1.0 --k2 0.0 0.5 --k3 1.5
python evaluate.py --dataset Cora --model GAT --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GCN --self_buff 10 --grad_norm 1 \
    --k1 1.0 --k2 0.8 0.2 --k3 2
python evaluate.py --dataset CiteSeer --model GCN --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GraphSAGE --self_buff 15 --grad_norm 1 \
    --k1 1.0 --k2 0 0.8 --k3 3
python evaluate.py --dataset CiteSeer --model GraphSAGE --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GAT --self_buff 15 --grad_norm 1 \
    --k1 1 --k2 0.8 0 --k3 4
python evaluate.py --dataset CiteSeer --model GAT --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GCN --self_buff 10 --grad_norm 1 \
    --k1 0.4 --k2 0.5 0.0 --k3 1
python evaluate.py --dataset PubMed --model GCN --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GraphSAGE --self_buff 10 --grad_norm 1 \
    --k1 0.2 --k2 0.9 0 --k3 1
python evaluate.py --dataset PubMed --model GraphSAGE --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GAT --self_buff 0 --grad_norm 1 \
    --k1 1 --k2 0.2 0 --k3 1
python evaluate.py --dataset PubMed --model GAT --val_ratio 0.2
