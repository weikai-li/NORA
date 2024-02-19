python main.py --dataset PubMed --method nora --model GCN --decay 0.95 --self_buff 20 --k3 0 \
    --k1 0.4 --k2 1.0 0.0
python evaluate.py --dataset PubMed --model GCN --val_ratio 0.2
# python main.py --dataset PubMed --method nora --model GCN --decay 0.95 --self_buff 30 --k3 0 \
#     --k1 0.4 --k2 1.0 0.0
# python evaluate.py --dataset PubMed --model GCN --val_ratio 0.2
# python main.py --dataset PubMed --method nora --model GCN --decay 0.95 --self_buff 40 --k3 0 \
#     --k1 0.4 --k2 1.0 0.0
# python evaluate.py --dataset PubMed --model GCN --val_ratio 0.2
