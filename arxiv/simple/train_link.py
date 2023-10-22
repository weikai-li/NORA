import argparse
import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, SAGEConv
from dgl.utils import expand_as_pair
import os


class MyGCNConv(GraphConv):
    def forward(self, graph, feat, weight=None, edge_weight=None, node_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight
            if node_weight is not None:
                feat_src = feat_src * node_weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCN_arxiv_edge(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_arxiv_edge, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(MyGCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(MyGCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(MyGCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, graph, x, return_hidden=False, edge_weight=None, node_weight=None):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            xs.append(x)
            x = conv(graph, x, edge_weight=edge_weight, node_weight=node_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
        x = self.convs[-1](graph, x, edge_weight=edge_weight, node_weight=node_weight)
        xs.append(x)
        if return_hidden:
            return x, xs
        else:
            return x


def gen_neg_link(num_neg, num_node, num_edge, edge_index):
    neg_idx1 = np.random.randint(0, num_node-1, num_neg)
    neg_idx2 = np.random.randint(0, num_node-1, num_neg)
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
    for i in range(len(neg_idx1)):
        if (neg_idx2[i] not in link_dict[neg_idx1[i]]) and (neg_idx1[i] not in link_dict[neg_idx2[i]]):
            link_dict[neg_idx1[i]].append(neg_idx2[i])
            link_dict[neg_idx2[i]].append(neg_idx1[i])
            neg_link_list.append([neg_idx1[i], neg_idx2[i]])
    neg_link = np.array(neg_link_list).T
    return neg_link


def main(cycle):
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data = DglNodePropPredDataset(name="ogbn-arxiv", root='../dataset')
    graph, labels = data[0]
    feat = graph.ndata['feat']
    num_node = len(feat)
    edge_index = [np.array(graph.edges()[0]), np.array(graph.edges()[1])]
    edge_index = np.array(edge_index)
    num_edge = edge_index.shape[1]

    try:
        test_neg = np.load(f'../dataset/ogbn_arxiv/processed/{cycle}_neg_link.npy')
    except:
        test_neg = gen_neg_link(int(0.3 * num_edge), num_node, num_edge, edge_index)
        np.save(f'../dataset/ogbn_arxiv/processed/{cycle}_neg_link.npy', test_neg)
    train_neg = gen_neg_link(int(2.4 * num_edge), num_node, num_edge, edge_index)
    val_neg = gen_neg_link(int(0.3 * num_edge), num_node, num_edge, edge_index)
    train_neg, val_neg, test_neg = torch.tensor(train_neg), torch.tensor(val_neg), torch.tensor(test_neg)

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
    
    train_link = torch.tensor(edge_index[:, train_mask])
    val_link = torch.tensor(edge_index[:, val_mask])
    test_link = torch.tensor(edge_index[:, test_mask])
    graph = dgl.to_bidirected(graph)
    graph.ndata['feat'] = feat
    graph = graph.remove_self_loop().add_self_loop()       # add self-loop
    graph.create_formats_()                                # create sparse matrics for all possible matrics
    graph, labels, train_link, val_link, test_link, train_neg, val_neg, test_neg = map(
        lambda x: x.to(device), (graph, labels, train_link, val_link, test_link, train_neg, val_neg, test_neg)
    )

    if args.model == 'GCN':
        model = GCN_arxiv_edge(graph.ndata["feat"].shape[1], args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout).to(device)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_acc = 0

        for epoch in range(1, 1 + args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(graph, graph.ndata['feat'])
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
                out = model(graph, graph.ndata['feat'])
                out_pos = out[val_link[0]] * out[val_link[1]]
                out_pos = out_pos.sum(1)
                out_neg = out[val_neg[0]] * out[val_neg[1]]
                out_neg = out_neg.sum(1)
                out_all = torch.concat([out_pos, out_neg])
                pred = (torch.sigmoid(out_all) > 0.5)
                labels = torch.concat([torch.ones_like(out_pos), torch.zeros_like(out_neg)])
                correct = (pred == labels).sum()
                acc = int(correct) / (len(labels))
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f'saved_model/{cycle}_gcn-edge_{args.num_layers}_{args.hidden_channels}.pkl')

            if epoch % args.log_steps == 0:
                print(f'Cycle: {cycle}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Valid: {100 * acc:.2f}% ')


if __name__ == "__main__":
    os.makedirs('saved_model', exist_ok=True)
    for cycle in range(5):
        main(cycle)
