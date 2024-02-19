python main.py --dataset PubMed --method nora --model GAT --decay 0.9 --self_buff 7 --k3 0 \
    --k1 0.5 --k2 0.4 0.2 --use_message
python evaluate.py --dataset PubMed --model GAT --val_ratio 0.2
# python main.py --dataset PubMed --method nora --model GAT --decay 0.95 --self_buff 7 --k3 0 \
#     --k1 0.5 --k2 0.4 0.2 --use_message
# python evaluate.py --dataset PubMed --model GAT --val_ratio 0.2
# python main.py --dataset PubMed --method nora --model GAT --decay 1.0 --self_buff 7 --k3 0 \
#     --k1 0.5 --k2 0.4 0.2 --use_message
# python evaluate.py --dataset PubMed --model GAT --val_ratio 0.2
