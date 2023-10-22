import argparse
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


class MySAGEConv(SAGEConv):
    def forward(self, graph, feat, edge_weight=None, node_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            assert self._aggre_type == "mean"
            if lin_before_mp:
                feat_src = self.fc_neigh(feat_src)
            if node_weight is not None:
                feat_src = feat_src * node_weight
            graph.srcdata["h"] = feat_src
            graph.update_all(msg_fn, fn.mean("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
            
            rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class GCN_arxiv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_arxiv, self).__init__()
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
            return x.log_softmax(dim=-1), xs
        else:
            return x.log_softmax(dim=-1)


class GraphSAGE_arxiv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GraphSAGE_arxiv, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(MySAGEConv(in_channels, hidden_channels, aggregator_type='mean'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(MySAGEConv(hidden_channels, hidden_channels, aggregator_type='mean'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(MySAGEConv(hidden_channels, out_channels, aggregator_type='mean'))
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
            return x.log_softmax(dim=-1), xs
        else:
            return x.log_softmax(dim=-1)


def train(model, graph, labels, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(graph, graph.ndata['feat'])
    loss = F.nll_loss(out[train_idx], labels.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, graph, labels, evaluator, train_idx, val_idx, test_idx):
    model.eval()

    out = model(graph, graph.ndata['feat'])
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': labels[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': labels[val_idx],
        'y_pred': y_pred[val_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc


def main(cycle):
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
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
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    feat = graph.ndata['feat']  
    graph = dgl.to_bidirected(graph)
    graph.ndata['feat'] = feat
    graph = graph.remove_self_loop().add_self_loop()       # add self-loop
    graph.create_formats_()                                # create sparse matrics for all possible matrics
    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )

    all_idx = torch.cat([train_idx, val_idx, test_idx])
    offset = len(all_idx) * (cycle) / 5
    offset = int(offset)
    train_idx1 = all_idx[offset: min(len(train_idx)+offset, len(all_idx))]
    train_idx2 = all_idx[max(0, offset-len(all_idx)): max(0, len(train_idx)+offset-len(all_idx))]
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

    if args.model == 'GCN':
        model = GCN_arxiv(graph.ndata["feat"].shape[1], args.hidden_channels,
                    40, args.num_layers, args.dropout).to(device)
    elif args.model == 'GraphSAGE':
        model = GraphSAGE_arxiv(graph.ndata["feat"].shape[1], args.hidden_channels,
                     40, args.num_layers, args.dropout).to(device)
    elif args.model == 'GAT':
        model = GAT_arxiv(graph.ndata["feat"].shape[1], args.hidden_channels,
                     40, args.num_layers, args.dropout, num_heads=3).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        best_val, final_test = 0, 0

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, graph, labels, train_idx, optimizer)
            train_acc, valid_acc, test_acc = test(model, graph, labels, evaluator, train_idx, val_idx, test_idx)

            if valid_acc > best_val:
                best_val = valid_acc
                final_test = test_acc
                saved_name = f'saved_model/{cycle}_{args.model.lower()}_{args.num_layers}_{args.hidden_channels}.pkl'
                torch.save(model.state_dict(), saved_name)

            if epoch % args.log_steps == 0:
                print(f'Cycle: {cycle}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}% '
                      f'Final Test: {100 * final_test:.2f}%')
        
        print(f'best val: {100 * best_val:.2f}, final test: {100 * final_test:.2f}')


if __name__ == "__main__":
    os.makedirs('saved_model', exist_ok=True)
    for cycle in range(5):
        main(cycle)
