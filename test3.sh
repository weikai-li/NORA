python main.py --dataset Cora --method nora --model GCN --decay 0.9 --self_buff 8 --k3 0 \
    --k1 1.0 --k2 0.5 0.0
python evaluate.py --dataset Cora --model GCN --val_ratio 0.2

# python main.py --dataset PubMed --method nora --model GraphSAGE --decay 1.0 --self_buff 30 --k3 0 \
#     --k1 0.2 --k2 1.0 0.0
# python evaluate.py --dataset PubMed --model GraphSAGE --val_ratio 0.2
# python main.py --dataset PubMed --method nora --model GraphSAGE --decay 1.0 --self_buff 30 --k3 0 \
#     --k1 0.2 --k2 1.0 0.0
# python evaluate.py --dataset PubMed --model GraphSAGE --val_ratio 0.2
# python main.py --dataset PubMed --method nora --model GraphSAGE --decay 1.0 --self_buff 30 --k3 0 \
#     --k1 0.2 --k2 1.0 0.0
# python evaluate.py --dataset PubMed --model GraphSAGE --val_ratio 0.2
