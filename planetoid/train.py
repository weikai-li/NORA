import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import roc_auc_score
import os
import copy
try:
    from models import MyGCN, MyGraphSAGE, MyGAT, MyGCNII
    from best_config import planetoid_config
except:
    from planetoid.models import MyGCN, MyGraphSAGE, MyGAT, MyGCNII
    from planetoid.best_config import planetoid_config


# Node classification, train:val:test = 50%:30%:20%
def cycle_node_split(g, cycle):
    num_node = g.num_nodes()
    train_mask = torch.zeros(num_node, dtype=bool)
    val_mask = torch.zeros(num_node, dtype=bool)
    test_mask = torch.zeros(num_node, dtype=bool)
    num_train = round(0.5 * num_node)
    num_val = round(0.3 * num_node)
    num_test = num_node - num_train - num_val
    offset = round(num_node * cycle / 5)
    train_mask[offset: min(offset+num_train, num_node)] = True
    train_mask[0: max(0, num_train+offset-num_node)] = True
    val_mask[min(num_train+offset, num_node): min(num_train+num_val+offset, num_node)] = True
    val_mask[max(0, num_train+offset-num_node): max(0, num_train+num_val+offset-num_node)] = True
    test_mask[min(num_train+num_val+offset, num_node): num_node] = True
    test_mask[max(0, offset-num_test): max(0, offset)] = True
    assert (train_mask.sum() + val_mask.sum() + test_mask.sum()) == num_node
    assert (train_mask & val_mask).sum() == 0
    assert (val_mask & test_mask).sum() == 0
    assert (train_mask & test_mask).sum() == 0
    return train_mask, val_mask, test_mask


# Link prediction, train:val:test = 80%:10%:10%. Positive:negative = 1:1.
def cycle_edge_split(g, cycle, args):
    edge_index = [g.edges()[0], g.edges()[1]]
    edge_index = torch.stack(edge_index)
    mask = (edge_index[1] >= edge_index[0])
    edge_index = edge_index[:, mask]
    num_edge = edge_index.shape[1]

    train_mask = torch.zeros(num_edge, dtype=bool)
    val_mask = torch.zeros(num_edge, dtype=bool)
    test_mask = torch.zeros(num_edge, dtype=bool)
    num_train = round(0.8 * num_edge)
    num_val = round(0.1 * num_edge)
    num_test = num_edge - num_train - num_val
    offset = round(cycle / 5 * num_edge)
    train_mask[offset: min(offset+num_train, num_edge)] = True
    train_mask[0: max(0, num_train+offset-num_edge)] = True
    val_mask[min(num_train+offset, num_edge): min(num_train+num_val+offset, num_edge)] = True
    val_mask[max(0, num_train+offset-num_edge): max(0, num_train+num_val+offset-num_edge)] = True
    test_mask[min(num_train+num_val+offset, num_edge): num_edge] = True
    test_mask[max(0, offset-num_test): max(0, offset)] = True
    assert (train_mask.sum() + val_mask.sum() + test_mask.sum()) == num_edge
    assert (train_mask & val_mask).sum() == 0
    assert (val_mask & test_mask).sum() == 0
    assert (train_mask & test_mask).sum() == 0
    train_link = edge_index[:, train_mask].clone().detach()
    val_link = edge_index[:, val_mask].clone().detach()
    test_link = edge_index[:, test_mask].clone().detach()

    num_node = g.num_nodes()
    os.makedirs(f'./data/{args.dataset}', exist_ok=True)
    try:
        train_neg = np.load(f'./data/{args.dataset}/{cycle}_train_neg_link.npy')
        val_neg = np.load(f'./data/{args.dataset}/{cycle}_val_neg_link.npy')
        test_neg = np.load(f'./data/{args.dataset}/{cycle}_neg_link.npy')
        train_neg = torch.tensor(train_neg)
        val_neg = torch.tensor(val_neg)
        test_neg = torch.tensor(test_neg)
    except:
        edge_index = np.array(edge_index)
        neg_idx1 = np.random.randint(0, num_node, size=round(1.1 * num_edge))
        neg_idx2 = np.random.randint(0, num_node, size=round(1.1 * num_edge))
        unequal_mask = (neg_idx1 != neg_idx2)
        neg_idx1 = list(neg_idx1[unequal_mask])
        neg_idx2 = list(neg_idx2[unequal_mask])
        neg_link_list = []
        link_dict = {}
        for i in range(num_node):
            link_dict[i] = []
        for i in range(num_edge):
            link_dict[edge_index[0, i]].append(edge_index[1, i])
            link_dict[edge_index[1, i]].append(edge_index[0, i])
        cnt = 0
        for i in range(len(neg_idx1)):
            if (neg_idx2[i] not in link_dict[neg_idx1[i]]) and (neg_idx1[i] not in link_dict[neg_idx2[i]]):
                link_dict[neg_idx1[i]].append(neg_idx2[i])
                link_dict[neg_idx2[i]].append(neg_idx1[i])
                neg_link_list.append([neg_idx1[i], neg_idx2[i]])
                cnt += 1
                if cnt == num_edge:
                    break
        neg_link = torch.tensor(neg_link_list).T
        train_neg = neg_link[:, : num_train]
        val_neg = neg_link[:, num_train: num_train + num_val]
        test_neg = neg_link[:, num_train + num_val:]
        assert test_neg.shape[1] == num_test
        np.save(f'./data/{args.dataset}/{cycle}_train_neg_link.npy', train_neg)
        np.save(f'./data/{args.dataset}/{cycle}_val_neg_link.npy', val_neg)
        np.save(f'./data/{args.dataset}/{cycle}_neg_link.npy', test_neg)
    return train_link, val_link, test_link, train_neg, val_neg, test_neg


def load_dataset(args, data_dir='./data'):
    if args.dataset == 'Cora':
        dataset = dgl.data.CoraGraphDataset(raw_dir=data_dir, verbose=False)
    elif args.dataset == 'CiteSeer':
        dataset = dgl.data.CiteseerGraphDataset(raw_dir=data_dir, verbose=False)
    else:
        assert args.dataset == 'PubMed'
        dataset = dgl.data.PubmedGraphDataset(raw_dir=data_dir, verbose=False)
    return dataset


def load_model(args, dataset):
    g = dataset[0]
    if args.model == 'GCN':
        model = MyGCN(g.ndata["feat"].shape[1], args.hidden_size, dataset.num_classes,
            args.num_layers, args.dropout)
    elif args.model == 'GraphSAGE':
        model = MyGraphSAGE(g.ndata["feat"].shape[1], args.hidden_size, dataset.num_classes,
            args.num_layers, args.dropout)
    elif args.model == 'GAT':
        model = MyGAT(g.ndata["feat"].shape[1], args.hidden_size, dataset.num_classes,
            args.num_layers, args.dropout)
    elif args.model == 'GCNII':
        model = MyGCNII(g.ndata["feat"].shape[1], args.hidden_size, dataset.num_classes,
            args.num_layers, args.dropout, alpha=args.gcnii_alpha, lambda_=args.gcnii_lambda)
    elif args.model == 'GCN_edge':
        model = MyGCN(g.ndata["feat"].shape[1], args.hidden_size, args.hidden_size,
            args.num_layers, args.dropout)
    elif args.model == 'GraphSAGE_edge':
        model = MyGraphSAGE(g.ndata["feat"].shape[1], args.hidden_size, args.hidden_size,
            args.num_layers, args.dropout)
    elif args.model == 'GAT_edge':
        model = MyGAT(g.ndata["feat"].shape[1], args.hidden_size, args.hidden_size,
            args.num_layers, args.dropout)
    else:
        assert args.model == 'GCNII_edge'
        model = MyGCNII(g.ndata["feat"].shape[1], args.hidden_size, args.hidden_size,
            args.num_layers, args.dropout, alpha=args.gcnii_alpha, lambda_=args.gcnii_lambda)
    return model


def train(cycle, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset(args)
    model = load_model(args, dataset)
    g = dataset[0]
    # Cycle the data split.
    if args.link_prediction == False:
        train_mask, val_mask, test_mask = cycle_node_split(g, cycle)
    else:
        train_link, val_link, test_link, train_neg, val_neg, test_neg = cycle_edge_split(g, cycle, args)

    model = model.to(device)
    if args.link_prediction == False:
        g, train_mask, val_mask, test_mask = map(
            lambda x: x.to(device), (g, train_mask, val_mask, test_mask)
        )
    else:
        g, train_link, val_link, test_link, train_neg, val_neg, test_neg = map(
            lambda x: x.to(device), (g, train_link, val_link, test_link, train_neg, val_neg, test_neg)
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_acc = 0
    best_test_acc = 0
    best_model = None
    features = g.ndata["feat"]
    labels = g.ndata["label"]

    for epoch in range(args.num_epochs):
        model.train()
        if args.link_prediction == False:
            logits = model(g, features)
            pred = logits.argmax(1)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_model = copy.deepcopy(model)
        else:
            out = model(g, features)
            out_pos = out[train_link[0]] * out[train_link[1]]
            out_pos = out_pos.sum(1)
            out_neg = out[train_neg[0]] * out[train_neg[1]]
            out_neg = out_neg.sum(1)
            out_all = torch.concat([out_pos, out_neg])
            labels = torch.concat([torch.ones_like(out_pos), torch.zeros_like(out_neg)])
            loss = F.binary_cross_entropy_with_logits(out_all, labels)
            with torch.no_grad():
                model.eval()
                val_out_pos = out[val_link[0]] * out[val_link[1]]
                val_out_pos = val_out_pos.sum(1)
                val_out_neg = out[val_neg[0]] * out[val_neg[1]]
                val_out_neg = val_out_neg.sum(1)
                val_out_all = torch.concat([val_out_pos, val_out_neg])
                val_pred = torch.sigmoid(val_out_all)
                val_labels = np.concatenate([np.ones(val_link.shape[1]), np.zeros(val_neg.shape[1])])
                val_acc = roc_auc_score(y_true=val_labels, y_score=val_pred.detach().cpu())
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    test_out_pos = out[test_link[0]] * out[test_link[1]]
                    test_out_pos = test_out_pos.sum(1)
                    test_out_neg = out[test_neg[0]] * out[test_neg[1]]
                    test_out_neg = test_out_neg.sum(1)
                    test_out_all = torch.concat([test_out_pos, test_out_neg])
                    test_pred = torch.sigmoid(test_out_all)
                    test_labels = np.concatenate([np.ones(test_link.shape[1]), np.zeros(test_neg.shape[1])])
                    test_acc = roc_auc_score(y_true=test_labels, y_score=test_pred.detach().cpu())
                    best_test_acc = test_acc
                    best_model = copy.deepcopy(model)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.log_steps == 0:
            print(
                f"Epoch {epoch}, loss: {loss:.3f}, val: {val_acc:.3f} (best {best_val_acc:.3f}), "
                f"test: {test_acc:.3f} (best {best_test_acc:.3f})"
            )
    
    saved_name = f'saved_model/{cycle}_{args.dataset}_{args.model.lower()}_{args.num_layers}'
    saved_name += f'_{args.hidden_size}.pkl'
    torch.save(best_model.state_dict(), saved_name)
    print('Saved GNN model at:', saved_name)
    print()
    return best_val_acc.item(), best_test_acc.item()


def load_config(args):
    config = planetoid_config[args.model][args.dataset]
    if args.num_layers == 0:
        args.num_layers = config['num_layers']
    if args.hidden_size == 0:
        args.hidden_size = config['hidden_size']
    if args.dropout == 0:
        args.dropout = config['dropout']
    if args.model in ['GCNII', 'GCNII_edge']:
        if hasattr(args, 'gcnii_alpha'):
            if args.gcnii_alpha == 0:
                args.gcnii_alpha = config['gcnii_alpha']
            if args.gcnii_lambda == 0:
                args.gcnii_lambda = config['gcnii_lambda']
        else:
            args.gcnii_alpha = config['gcnii_alpha']
            args.gcnii_lambda = config['gcnii_lambda']
    return args


def main():
    parser = argparse.ArgumentParser(description='Planetoid')
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT', 'GCNII'])
    # 0 means using the default tuned hyper-parameters
    parser.add_argument('--num_layers', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--gcnii_alpha', type=float, default=0)
    parser.add_argument('--gcnii_lambda', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--link_prediction', action='store_true', default=False)
    args = parser.parse_args()

    if args.link_prediction == True:
        args.model = f'{args.model}_edge'
    args = load_config(args)
    config = planetoid_config[args.model][args.dataset]
    if args.lr == 0:
        args.lr = config['lr']
    if args.wd == 0:
        args.wd = config['wd']
    print(args)

    val_accs, test_accs = [], []
    for cycle in range(5):
        val_acc, test_acc = train(cycle, args)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    print(f'Average val: {np.mean(val_accs):.4f}, test: {np.mean(test_accs):.4f}')


if __name__ == '__main__':
    os.makedirs('saved_model', exist_ok=True)
    main()
