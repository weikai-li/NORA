import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import degree
from models import MyGCN, MyGraphSAGE, MyGAT
import argparse
from best_config import config
import copy

argparser = argparse.ArgumentParser("Bias Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--dataset", type=str, default="Cora", choices=['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv'])
argparser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT', 'DrGAT'])
argparser.add_argument('--n_layers', type=int, default=0, help="0: use default")
argparser.add_argument('--hidden_size', type=int, default=0, help="0: use default")
args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = config[args.model][args.dataset]

dataset = Planetoid(name=args.dataset, root='./data')
num_features = dataset.num_features
num_classes = dataset.num_classes

if args.n_layers == 0:
    args.n_layers = config['n_layers']
if args.hidden_size == 0:
    args.hidden_size = config['hidden_size']
args.lr = config['lr']
if args.model == 'GCN':
    model = MyGCN(in_channels=num_features, out_channels=num_classes, hidden_channels=args.hidden_size, 
        num_layers=args.n_layers, dropout=config['dropout'])
elif args.model == 'GraphSAGE':
    model = MyGraphSAGE(in_channels=num_features, out_channels=num_classes, hidden_channels=args.hidden_size, 
        num_layers=args.n_layers, dropout=config['dropout'])
elif args.model == 'GAT':
    model = MyGAT(in_channels=num_features, out_channels=num_classes, hidden_channels=args.hidden_size, 
        num_layers=args.n_layers, dropout=config['dropout'])
model = model.to(device)


def train(cycle):
    dataset = Planetoid(name=args.dataset, root='./data')
    dataset.transform = T.NormalizeFeatures()
    data = dataset.transform(dataset[0])
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    num_node = data.x.shape[0]

    # train:val:test = 50%:30%:20%
    train_mask = torch.zeros(num_node, dtype=bool)
    val_mask = torch.zeros(num_node, dtype=bool)
    test_mask = torch.zeros(num_node, dtype=bool)
    num_train = int(0.5 * num_node)
    num_val = int(0.3 * num_node)
    num_test = num_node - num_train - num_val

    offset = int(cycle / 5 * num_node)
    train_mask[offset: min(offset+num_train, num_node)] = True
    train_mask[max(0, offset-num_node): max(0, num_train+offset-num_node)] = True
    val_mask[min(num_train+offset, num_node): min(num_train+num_val+offset, num_node)] = True
    val_mask[max(0, num_train+offset-num_node): max(0, num_train+num_val+offset-num_node)] = True
    test_mask[min(num_train+num_val+offset, num_node): num_node] = True
    test_mask[max(0, offset-num_test): max(0, offset)] = True

    assert (train_mask.sum() + val_mask.sum() + test_mask.sum()) == num_node
    assert (train_mask & val_mask).sum() == 0
    assert (val_mask & test_mask).sum() == 0
    assert (train_mask & test_mask).sum() == 0

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config['wd'])
    best_acc, patience = 0, 0
    best_model = None

    for epoch in range(config['n_epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = F.softmax(out, dim=1)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            acc = int(correct) / int(data.val_mask.sum())
            if acc > best_acc:
                best_acc = acc
                print(epoch, acc)
                # final_out = F.softmax(out, dim=1)
                if args.hidden_size == config['hidden_size']:
                    torch.save(model.state_dict(), f'saved_model/{args.dataset}_{args.model}_{args.n_layers}_{cycle}.pkl')
                else:
                    torch.save(model.state_dict(), f'saved_model/{args.dataset}_{args.model}_{args.n_layers}_{args.hidden_size}_{cycle}.pkl')
            else:
                if patience == config['patience']:
                    break
                else:
                    patience = patience + 1


if __name__ == '__main__':
    for cycle in range(5):
        train(cycle)
