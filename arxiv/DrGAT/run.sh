# We remove the label propagation process in DrGAT's original script, so that our code can 
# make inference purely based on the learned propagation patterns, instead of training-set labels

# The following codes will train the model with cycle=0. Please manually set cycle to 0, 1, 2, 3, 4
# to cycle the data split and train the GNN model for 5 times

# First train the teacher model:
python main.py --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 0 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 256 --save kd --backbone drgat --mode teacher
# After the teacher model is trained, train the student model:
python main.py --gpu 0 --use-norm --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --cycle 0 \
    --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 256 --save kd --backbone drgat --alpha 0.95 --temp 0.7 --mode student
