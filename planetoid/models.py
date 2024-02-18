import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, SAGEConv, GATConv
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair


class GNN_basic(torch.nn.Module):
    def __init__(self, dropout, arxiv_data):
        super(GNN_basic, self).__init__()
        self.arxiv_data = arxiv_data
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.arxiv_data == True:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, graph, x, return_hidden=False, edge_weight=None, node_weight=None):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            xs.append(x)
            x = conv(graph, x, edge_weight=edge_weight, node_weight=node_weight)
            if isinstance(self, MyGAT):
                x = x.reshape(x.shape[0], -1)
            if self.arxiv_data == True:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
        x = self.convs[-1](graph, x, edge_weight=edge_weight, node_weight=node_weight)
        if isinstance(self, MyGAT):
            x = x.reshape(x.shape[0], -1)
        xs.append(x)
        if return_hidden:
            return x, xs
        else:
            return x


class MyGCNConv(GraphConv):
    def forward(self, graph, feat, weight=None, edge_weight=None, node_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    pass
                    # raise DGLError(
                    #     "There are 0-in-degree nodes in the graph, "
                    #     "output for those nodes will be invalid. "
                    #     "This is harmful for some applications, "
                    #     "causing silent performance regression. "
                    #     "Adding self-loop on the input graph by "
                    #     "calling `g = dgl.add_self_loop(g)` will resolve "
                    #     "the issue. Setting ``allow_zero_in_degree`` "
                    #     "to be `True` when constructing this module will "
                    #     "suppress the check and let the code run."
                    # )
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

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                if node_weight is not None:
                    feat_src = feat_src * node_weight
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                if node_weight is not None:
                    feat_src = feat_src * node_weight
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


class MyGCN(GNN_basic):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, arxiv_data=False):
        super(MyGCN, self).__init__(dropout, arxiv_data)
        self.convs = torch.nn.ModuleList()
        self.convs.append(MyGCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(MyGCNConv(hidden_channels, hidden_channels))
        self.convs.append(MyGCNConv(hidden_channels, out_channels))
        if arxiv_data == True:
            self.bns = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.reset_parameters()


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


class MyGraphSAGE(GNN_basic):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, arxiv_data=False):
        super(MyGraphSAGE, self).__init__(dropout, arxiv_data)
        self.convs = torch.nn.ModuleList()
        self.convs.append(MySAGEConv(in_channels, hidden_channels, aggregator_type='mean'))
        for _ in range(num_layers - 2):
            self.convs.append(MySAGEConv(hidden_channels, hidden_channels, aggregator_type='mean'))
        self.convs.append(MySAGEConv(hidden_channels, out_channels, aggregator_type='mean'))
        if arxiv_data == True:
            self.bns = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.reset_parameters()


class MyGATConv(GATConv):
    def forward(self, graph, feat, edge_weight=None, get_attention=False, node_weight=None):
        assert edge_weight == None and node_weight == None
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    pass
                    # raise DGLError(
                    #     "There are 0-in-degree nodes in the graph, "
                    #     "output for those nodes will be invalid. "
                    #     "This is harmful for some applications, "
                    #     "causing silent performance regression. "
                    #     "Adding self-loop on the input graph by "
                    #     "calling `g = dgl.add_self_loop(g)` will resolve "
                    #     "the issue. Setting ``allow_zero_in_degree`` "
                    #     "to be `True` when constructing this module will "
                    #     "suppress the check and let the code run."
                    # )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            if node_weight is not None:
                feat_src = feat_src * node_weight
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            try:
                if self.has_explicit_bias:
                    rst = rst + self.bias.view(
                        *((1,) * len(dst_prefix_shape)),
                        self._num_heads,
                        self._out_feats
                    )
            except:    # Old version dgl
                if self.bias is not None:
                    rst = rst + self.bias.view(
                        *((1,) * len(dst_prefix_shape)),
                        self._num_heads,
                        self._out_feats
                    )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class MyGAT(GNN_basic):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, arxiv_data=False):
        super(MyGAT, self).__init__(dropout, arxiv_data)
        self.convs = torch.nn.ModuleList()
        if arxiv_data == False:
            num_heads = 2
        else:
            num_heads = 3
        self.convs.append(MyGATConv(in_channels, hidden_channels, num_heads=num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(MyGATConv(num_heads * hidden_channels, hidden_channels, num_heads=num_heads))
        self.convs.append(MyGATConv(num_heads * hidden_channels, out_channels, num_heads=1))
        if arxiv_data == True:
            self.bns = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * num_heads))
        self.reset_parameters()
