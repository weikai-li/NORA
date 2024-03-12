import argparse
import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import copy
from sklearn.metrics import roc_auc_score
import os
import sys
from os import path
sys.path.append(path.abspath('../../planetoid'))
try:
    from models import MyGCN, MyGraphSAGE, MyGAT, MyGCNII
    from best_config import arxiv_config
except:
    from planetoid.models import MyGCN, MyGraphSAGE, MyGAT, MyGCNII
    from arxiv.others.best_config import arxiv_config


def cycle_node_split(data, cycle):
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    offset = round(len(all_idx) * (cycle) / 5)
    train_idx1 = all_idx[offset: min(len(train_idx)+offset, len(all_idx))]
    train_idx2 = all_idx[0: max(0, len(train_idx)+offset-len(all_idx))]
    assert len(train_idx1) + len(train_idx2) == len(train_idx)
    train_idx = torch.cat([train_idx1, train_idx2])
    val_idx1 = all_idx[min(len(train_idx)+offset, len(all_idx)): min(len(train_idx)+len(val_idx)+offset, len(all_idx))]
    val_idx2 = all_idx[max(0, len(train_idx)+offset-len(all_idx)): max(0, len(train_idx)+len(val_idx)+offset-len(all_idx))]
    assert len(val_idx1) + len(val_idx2) == len(val_idx)
    val_idx = torch.cat([val_idx1, val_idx2])
    test_idx1 = all_idx[min(len(train_idx)+len(val_idx)+offset, len(all_idx)): len(all_idx)]
    test_idx2 = all_idx[max(0, offset-len(test_idx)): max(0, offset)]
    assert len(test_idx1) + len(test_idx2) == len(test_idx)
    test_idx = torch.cat([test_idx1, test_idx2])
    graph, labels = data[0]
    assert len(train_idx) + len(val_idx) + len(test_idx) == graph.num_nodes()
    train_set = set([i.item() for i in train_idx])
    val_set = set([i.item() for i in val_idx])
    test_set = set([i.item() for i in test_idx])
    assert len(train_set) == len(train_idx)
    assert len(val_set) == len(val_idx)
    assert len(test_set) == len(test_idx)
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    return train_idx, val_idx, test_idx


# Link prediction, train:val:test = 80%:10%:10%. Positive:negative=1:1.
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
    try:
        train_neg = np.load(f'../data/ogbn_arxiv/processed/{cycle}_train_neg_link.npy')
        val_neg = np.load(f'../data/ogbn_arxiv/processed/{cycle}_val_neg_link.npy')
        test_neg = np.load(f'../data/ogbn_arxiv/processed/{cycle}_neg_link.npy')
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
        np.save(f'../data/ogbn_arxiv/processed/{cycle}_train_neg_link.npy', train_neg)
        np.save(f'../data/ogbn_arxiv/processed/{cycle}_val_neg_link.npy', val_neg)
        np.save(f'../data/ogbn_arxiv/processed/{cycle}_neg_link.npy', test_neg)
    return train_link, val_link, test_link, train_neg, val_neg, test_neg


def load_dataset(data_dir='../data'):
    data = DglNodePropPredDataset(name="ogbn-arxiv", root=data_dir)
    return data


def load_model(args, graph):
    if args.model == 'GCN':
        model = MyGCN(graph.ndata["feat"].shape[1], args.hidden_size,
                    40, args.num_layers, args.dropout, arxiv_data=True)
    elif args.model == 'GraphSAGE':
        model = MyGraphSAGE(graph.ndata["feat"].shape[1], args.hidden_size,
                     40, args.num_layers, args.dropout, arxiv_data=True)
    elif args.model == 'GAT':
        model = MyGAT(graph.ndata["feat"].shape[1], args.hidden_size,
                     40, args.num_layers, args.dropout, arxiv_data=True)
    elif args.model == 'GCNII':
        model = MyGCNII(graph.ndata["feat"].shape[1], args.hidden_size, 40, args.num_layers,
            args.dropout, alpha=args.gcnii_alpha, lambda_=args.gcnii_lambda, arxiv_data=True)
    elif args.model == 'GCN_edge':
        model = MyGCN(graph.ndata["feat"].shape[1], args.hidden_size,
                     args.hidden_size, args.num_layers, args.dropout, arxiv_data=True)
    elif args.model == 'GraphSAGE_edge':
        model = MyGraphSAGE(graph.ndata["feat"].shape[1], args.hidden_size,
                     args.hidden_size, args.num_layers, args.dropout, arxiv_data=True)
    elif args.model == 'GAT_edge':
        model = MyGAT(graph.ndata["feat"].shape[1], args.hidden_size,
                     args.hidden_size, args.num_layers, args.dropout, arxiv_data=True)
    else:
        assert args.model == 'GCNII_edge'
        model = MyGCNII(graph.ndata["feat"].shape[1], args.hidden_size, args.hidden_size, args.num_layers,
            args.dropout, alpha=args.gcnii_alpha, lambda_=args.gcnii_lambda, arxiv_data=True)
    return model


def train(cycle, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_dataset()
    graph, labels = data[0]
    model = load_model(args, graph)
    # Cycle the data split.
    if args.link_prediction == False:
        train_idx, val_idx, test_idx = cycle_node_split(data, cycle)
    else:
        train_link, val_link, test_link, train_neg, val_neg, test_neg = cycle_edge_split(graph, cycle, args)

    feat = graph.ndata['feat']  
    graph = dgl.to_bidirected(graph)
    graph.ndata['feat'] = feat
    graph = graph.remove_self_loop().add_self_loop()       # add self-loop
    graph.create_formats_()                                # create sparse matrics for all possible matrics
    if args.link_prediction == False:
        graph, labels, train_idx, val_idx, test_idx = map(
            lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
        )
    else:
        graph, labels, train_link, val_link, test_link, train_neg, val_neg, test_neg = map(
            lambda x: x.to(device), (graph, labels, train_link, val_link, test_link, train_neg, val_neg, test_neg)
        )
    model = model.to(device)
    evaluator = Evaluator(name='ogbn-arxiv')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val, final_test = 0, 0

    for epoch in range(args.num_epochs):
        model.train()
        if args.link_prediction == False:
            out = model(graph, graph.ndata['feat'])
            loss = F.cross_entropy(out[train_idx], labels.squeeze(1)[train_idx])
            model.eval()
            y_pred = out.argmax(dim=-1, keepdim=True)
            valid_acc = evaluator.eval({
                'y_true': labels[val_idx],
                'y_pred': y_pred[val_idx],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': labels[test_idx],
                'y_pred': y_pred[test_idx],
            })['acc']
            if valid_acc > best_val:
                best_val = valid_acc
                final_test = test_acc
                best_model = copy.deepcopy(model)
        else:
            out = model(graph, graph.ndata['feat'])
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
                valid_acc = roc_auc_score(y_true=val_labels, y_score=val_pred.detach().cpu())
                if best_val < valid_acc:
                    best_val = valid_acc
                    test_out_pos = out[test_link[0]] * out[test_link[1]]
                    test_out_pos = test_out_pos.sum(1)
                    test_out_neg = out[test_neg[0]] * out[test_neg[1]]
                    test_out_neg = test_out_neg.sum(1)
                    test_out_all = torch.concat([test_out_pos, test_out_neg])
                    test_pred = torch.sigmoid(test_out_all)
                    test_labels = np.concatenate([np.ones(test_link.shape[1]), np.zeros(test_neg.shape[1])])
                    test_acc = roc_auc_score(y_true=test_labels, y_score=test_pred.detach().cpu())
                    final_test = test_acc
                    best_model = copy.deepcopy(model)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.log_steps == 0:
            print(
                f"Epoch {epoch}, loss: {loss:.3f}, val: {valid_acc:.3f} (best {best_val:.3f}), "
                f"test: {test_acc:.3f} (best {final_test:.3f})"
            )

    saved_name = f'./saved_model/{cycle}_{args.model.lower()}_{args.num_layers}_{args.hidden_size}.pkl'
    torch.save(best_model.state_dict(), saved_name)
    print('Saved GNN model at:', saved_name)
    print()
    return best_val, final_test


def load_config(args):
    if args.model == 'DrGAT':
        if args.num_layers == 0:
            args.num_layers = 3
        if args.hidden_size == 0:
            args.hidden_size = 256
        if args.dropout == 0:
            args.dropout = 0.8
    
    else:
        config = arxiv_config[args.model]
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
    parser = argparse.ArgumentParser(description='OGBN-Arxiv')
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--model', type=str, default='GCNII', choices=['GCN', 'GraphSAGE', 'GAT', 'GCNII'])
    # 0 means using the default best hyper-parameters
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
    config = arxiv_config[args.model]
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
