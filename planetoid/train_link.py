import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models import MyGCN
import argparse
from best_config import config
import os

argparser = argparse.ArgumentParser("Bias Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--dataset", type=str, default="Cora", choices=['Cora', 'CiteSeer', 'PubMed'])
argparser.add_argument("--model", type=str, default="GCN", choices=['GCN'])
argparser.add_argument('--n_layers', type=int, default=0, help="0: use default")
argparser.add_argument('--hidden_size', type=int, default=128, help="0: use default")
argparser.add_argument('--lr', type=float, default=1e-3)
args = argparser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = config[args.model][args.dataset]

dataset = Planetoid(name=args.dataset, root='./data')
num_features = dataset.num_features


if args.n_layers == 0:
    args.n_layers = config['n_layers']
if args.hidden_size == 0:
    args.hidden_size = config['hidden_size']
args.lr = config['lr']
if args.model == 'GCN':
    model = MyGCN(in_channels=num_features, out_channels=args.hidden_size, hidden_channels=args.hidden_size, 
        num_layers=args.n_layers, dropout=config['dropout'])
model = model.to(device)


def train(cycle):
    dataset = Planetoid(name=args.dataset, root='./data')
    dataset.transform = T.NormalizeFeatures()
    data = dataset.transform(dataset[0])
    edge_index = data.edge_index
    num_edge = edge_index.shape[1]

    # train:val:test = 80%:10%:10%
    train_mask = torch.zeros(num_edge, dtype=bool)
    val_mask = torch.zeros(num_edge, dtype=bool)
    test_mask = torch.zeros(num_edge, dtype=bool)
    num_train = int(0.8 * num_edge)
    num_val = int(0.1 * num_edge)
    num_test = num_edge - num_train - num_val
    offset = int(cycle / 5 * num_edge)
    train_mask[offset: min(offset+num_train, num_edge)] = True
    train_mask[max(0, offset-num_edge): max(0, num_train+offset-num_edge)] = True
    val_mask[min(num_train+offset, num_edge): min(num_train+num_val+offset, num_edge)] = True
    val_mask[max(0, num_train+offset-num_edge): max(0, num_train+num_val+offset-num_edge)] = True
    test_mask[min(num_train+num_val+offset, num_edge): num_edge] = True
    test_mask[max(0, offset-num_test): max(0, offset)] = True
    assert (train_mask.sum() + val_mask.sum() + test_mask.sum()) == num_edge
    assert (train_mask & val_mask).sum() == 0
    assert (val_mask & test_mask).sum() == 0
    assert (train_mask & test_mask).sum() == 0
    data = data.to(device)

    try:
        neg_link = np.load(f'data/{args.dataset}/processed/{cycle}_neg_link.npy')
    except:
        edge_index = edge_index.cpu().numpy()
        neg_idx1 = np.random.randint(0, len(data.x)-1, 3 * test_mask.sum().item())
        neg_idx2 = np.random.randint(0, len(data.x)-1, 3 * test_mask.sum().item())
        unequal_mask = (neg_idx1 != neg_idx2)
        neg_idx1 = list(neg_idx1[unequal_mask])
        neg_idx2 = list(neg_idx2[unequal_mask])
        neg_link_list = []
        link_dict = {}
        for i in range(len(data.x)):
            link_dict[i] = []
        for i in range(num_edge):
            link_dict[edge_index[0, i]].append(edge_index[1, i])
            link_dict[edge_index[1, i]].append(edge_index[0, i])
        for i in range(len(neg_idx1)):
            if (neg_idx2[i] not in link_dict[neg_idx1[i]]) and (neg_idx1[i] not in link_dict[neg_idx2[i]]):
                link_dict[neg_idx1[i]].append(neg_idx2[i])
                link_dict[neg_idx2[i]].append(neg_idx1[i])
                neg_link_list.append([neg_idx1[i], neg_idx2[i]])
        neg_link = np.array(neg_link_list).T
        np.save(f'data/{args.dataset}/processed/{cycle}_neg_link.npy', neg_link)
    
    edge_index = torch.tensor(edge_index).to(device)
    neg_link = torch.tensor(neg_link).to(device)
    train_link = edge_index[:, train_mask]
    val_link = edge_index[:, val_mask]
    test_link = edge_index[:, test_mask]
    train_neg = neg_link[:, :int(0.8 * neg_link.shape[1])]
    val_neg = neg_link[:, int(0.8 * neg_link.shape[1]):int(0.9 * neg_link.shape[1])]
    test_neg = neg_link[:, int(0.9 * neg_link.shape[1]):]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config['wd'])
    best_acc = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, train_link)
        out_pos = out[train_link[0]] * out[train_link[1]]
        out_pos = out_pos.sum(1)
        out_neg = out[train_neg[0]] * out[train_neg[1]]
        out_neg = out_neg.sum(1)
        out_all = torch.concat([out_pos, out_neg])
        labels = torch.concat([torch.ones_like(out_pos), torch.zeros_like(out_neg)])
        loss = F.binary_cross_entropy_with_logits(out_all, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, train_link)
            out_pos = out[val_link[0]] * out[val_link[1]]
            out_pos = out_pos.sum(1)
            out_neg = out[val_neg[0]] * out[val_neg[1]]
            out_neg = out_neg.sum(1)
            out_all = torch.concat([out_pos, out_neg])
            labels = torch.concat([torch.ones_like(out_pos), torch.zeros_like(out_neg)])
            pred = (torch.sigmoid(out_all) > 0.5)
            correct = (pred == labels).sum()
            acc = int(correct) / (len(val_link[0]) + len(val_neg[0]))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f'saved_model/{args.dataset}_{args.model}-edge_{args.n_layers}_{args.hidden_size}_{cycle}.pkl')
        if epoch % 10 == 9:
            print(f'Epoche: {epoch + 1}, Loss: {loss:.4f}, Valid acc: {100*acc:.2f}, Best Valid acc: {100*best_acc:.2f}')


if __name__ == '__main__':
    os.makedirs('saved_model', exist_ok = True) 
    for cycle in range(5):
        train(cycle)
