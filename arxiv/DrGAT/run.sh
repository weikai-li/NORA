# We remove the label propagation process in DrGAT's original script, so that our code can 
# make inference purely based on the learned propagation patterns, instead of training-set labels
python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 0 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --mode teacher

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 0 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --alpha 0.95 --temp 0.7 --mode student

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 1 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --mode teacher

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 1 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --alpha 0.95 --temp 0.7 --mode student

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 2 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --mode teacher

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 2 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --alpha 0.95 --temp 0.7 --mode student

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 3 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --mode teacher

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 3 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --alpha 0.95 --temp 0.7 --mode student

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 4 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --mode teacher

python main.py --pretrain_path ../dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 4 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 128 --save kd --backbone drgat --alpha 0.95 --temp 0.7 --mode student