python main.py --dataset Cora --method nora --model GCN_edge --self_buff 2 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 1
python evaluate.py --dataset Cora --model GCN_edge --val_ratio 0.2
python main.py --dataset Cora --method nora --model GraphSAGE_edge --self_buff 0 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 3
python evaluate.py --dataset Cora --model GraphSAGE_edge --val_ratio 0.2
python main.py --dataset Cora --method nora --model GAT_edge --self_buff 0 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 1
python evaluate.py --dataset Cora --model GAT_edge --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GCN_edge --self_buff 20 --grad_norm 2 \
    --k1 0.8 --k2 0 0 --k3 5
python evaluate.py --dataset CiteSeer --model GCN_edge --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GraphSAGE_edge --self_buff 0.1 --grad_norm 2 \
    --k1 1 --k2 0 0 --k3 3
python evaluate.py --dataset CiteSeer --model GraphSAGE_edge --val_ratio 0.2
python main.py --dataset CiteSeer --method nora --model GAT_edge --self_buff 0.1 --k3 5 \
    --k1 1 --k2 0 0
python evaluate.py --dataset CiteSeer --model GAT_edge --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GCN_edge --self_buff 2 --grad_norm 1 \
    --k1 1 --k2 0.8 0 --k3 3
python evaluate.py --dataset PubMed --model GCN_edge --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GraphSAGE_edge --self_buff 0 --grad_norm 2 \
    --k1 1 --k2 0.5 0 --k3 3
python evaluate.py --dataset PubMed --model GraphSAGE_edge --val_ratio 0.2
python main.py --dataset PubMed --method nora --model GAT_edge --self_buff 3 --grad_norm 2 \
    --k1 1 --k2 0.8 0 --k3 1.5
python evaluate.py --dataset PubMed --model GAT_edge --val_ratio 0.2
