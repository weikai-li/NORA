import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GAT
import argparse
from tqdm import tqdm
import dgl
import dgl.function as fn
from load import load_model
from planetoid.best_config import config as config_ori


def load_results(method, args):
    if method not in ['degree', 'betweenness']:
        save_name = f'./save/{args.dataset}_{args.model}_{method}'
        if args.model not in ['TIMME', 'TIMME-edge']:
            save_name += f'_{args.n_layers}'
            if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
                config = config_ori[args.model][args.dataset]
                args.hidden_size = config['hidden_size']
            save_name += f'_{args.hidden_size}'
        if method == 'brute':
            res = np.load(f'{save_name}_mean.npy', allow_pickle=True)
        else:
            res = np.load(f'{save_name}.npy', allow_pickle=True)
        assert len(res) == 5
        return res.mean(0)


def node_mask(run_time, args, device, config, graph_ori):
    if args.model == 'GCN-edge':
        if args.dataset == 'ogbn-arxiv':
            graph_ori, test_link, neg_link = graph_ori
        else:
            graph_ori, neg_link = graph_ori
    model = load_model(run_time, args, device, config, graph_ori)
    model.eval()
    if args.dataset == 'ogbn-arxiv':
        feat = graph_ori.ndata['feat']
        with torch.no_grad():
            ori_out = model(graph_ori, feat)
            if args.model == 'DrGAT':
                ori_out = F.softmax(ori_out, dim=1)
            elif args.model in ['GCN', 'GraphSAGE']:
                ori_out = torch.exp(ori_out)
            else:
                ori_out_pos = ori_out[test_link[0]] * ori_out[test_link[1]]
                ori_out_pos = ori_out_pos.sum(1)
                ori_out_pos = torch.sigmoid(ori_out_pos)
                ori_out_neg = ori_out[neg_link[0]] * ori_out[neg_link[1]]
                ori_out_neg = ori_out_neg.sum(1)
                ori_out_neg = torch.sigmoid(ori_out_neg)
                ori_out = torch.concat([ori_out_pos, ori_out_neg])
        num_node = graph_ori.num_nodes()
    elif args.dataset in ['P50', 'P_20_50']:
        features_ori, adjs_ori, triplets_ori = graph_ori
        features_ori = features_ori.to_dense()
        with torch.no_grad():
            if args.model == 'TIMME':
                ori_out = model(features_ori, adjs_ori, only_classify=True)
                ori_out = torch.exp(ori_out)
            elif args.model == 'TIMME-edge':
                ori_emb = model(features_ori, adjs_ori)
                ori_out = model.calc_score_by_relation_2(triplets_ori, ori_emb[:-1], cuda=True)
                ori_out = torch.cat(ori_out)
        num_node = len(features_ori)
    else:    # plantoid dataset
        with torch.no_grad():
            ori_out = model(graph_ori.x, graph_ori.edge_index)
            if args.model != 'GCN-edge':
                ori_out = F.softmax(ori_out, dim=1)
            else:
                ori_out_pos = ori_out[graph_ori.edge_index[0]] * ori_out[graph_ori.edge_index[1]]
                ori_out_pos = ori_out_pos.sum(1)
                ori_out_pos = torch.sigmoid(ori_out_pos)
                ori_out_neg = ori_out[neg_link[0]] * ori_out[neg_link[1]]
                ori_out_neg = ori_out_neg.sum(1)
                ori_out_neg = torch.sigmoid(ori_out_neg)
                ori_out = torch.concat([ori_out_pos, ori_out_neg])
        num_node = graph_ori.x.shape[0]
    weight = torch.ones((num_node, 1), device=device)
    weight.requires_grad = False
    
    optimizer = torch.optim.Adam([weight], lr=args.lr, weight_decay=0)
    model.train()
    for m in model.parameters():
        m.requires_grad = True

    for epoch in tqdm(range(args.n_epochs)):
        optimizer.zero_grad()
        if args.dataset == 'ogbn-arxiv':
            feat = graph_ori.ndata['feat']
            out = model(graph_ori, feat, node_weight=weight)
            if args.model == 'DrGAT':
                out = F.softmax(out, dim=1)
            elif args.model in ['GCN', 'GraphSAGE']:
                out = torch.exp(out)
            else:
                out_pos = out[test_link[0]] * out[test_link[1]]
                out_pos = out_pos.sum(1)
                out_pos = torch.sigmoid(out_pos)
                out_neg = out[neg_link[0]] * out[neg_link[1]]
                out_neg = out_neg.sum(1)
                out_neg = torch.sigmoid(out_neg)
                out = torch.concat([out_pos, out_neg])
        elif args.dataset in ['P50', 'P_20_50']:
            if args.model == 'TIMME':
                out = model(features_ori, adjs_ori, only_classify=True, node_weight=weight)
                out = torch.exp(out)
            elif args.model == 'TIMME-edge':
                emb = model(features_ori, adjs_ori, node_weight=weight)
                out = model.calc_score_by_relation_2(triplets_ori, emb[:-1], cuda=True)
                out = torch.concat(out)
        else:
            out = model(graph_ori.x, graph_ori.edge_index, node_weight=weight)
            if args.model != 'GCN-edge':
                out = F.softmax(out, dim=1)
            else:
                out_pos = out[graph_ori.edge_index[0]] * out[graph_ori.edge_index[1]]
                out_pos = out_pos.sum(1)
                out_pos = torch.sigmoid(out_pos)
                out_neg = out[neg_link[0]] * out[neg_link[1]]
                out_neg = out_neg.sum(1)
                out_neg = torch.sigmoid(out_neg)
                out = torch.concat([out_pos, out_neg])
        
        loss1 = - (out - ori_out).abs().mean()
        loss2 = (num_node - weight.sum()) / num_node
        loss = loss1 + args.alpha * loss2
        loss.backward()
        optimizer.step()
        weight.requires_grad = False
        weight[weight < 0.0] = 0.0
        weight[weight > 1.0] = 1.0
        weight.requires_grad = True

    new_bias_list = (1 - weight.detach()).cpu().numpy().squeeze(1)
    return new_bias_list


def gnn_predict(run_time, args, device, config, graph_ori):
    gt = load_results('brute', args)
    gt = torch.tensor(gt, dtype=torch.float32).to(device).unsqueeze(1)
    gt = (gt - gt.mean()) / gt.std() - gt.min()
    save_name = f'./save_grad/{args.dataset}_{args.model}_gradient'
    if args.model not in ['TIMME', 'TIMME-edge']:
        save_name += f'_{args.n_layers}_{args.hidden_size}'
    
    if args.dataset == 'ogbn-arxiv':
        if args.model == 'GCN-edge':
            graph_ori, test_link, neg_link = graph_ori
        input_feature = graph_ori.ndata['feat']
        pyg_graph = Data()
        pyg_graph.edge_index = torch.stack(graph_ori.edges(), dim=0)
        for attr, value in graph_ori.ndata.items():
            pyg_graph[attr] = value
        for attr, value in graph_ori.edata.items():
            pyg_graph[attr] = value
        edge_index = pyg_graph.edge_index
    elif args.dataset in ['P50', 'P_20_50']:
        input_feature, adjs, triplets = graph_ori
        edge_index = [adjs[i].coalesce().indices() for i in range(10)]
        edge_index = torch.concat(edge_index, 1)
        edge_index = edge_index.to(device)
    else:    # plantoid dataset
        if args.model == 'GCN-edge':
            graph_ori, neg_link = graph_ori
        input_feature = graph_ori.x
        edge_index = graph_ori.edge_index
    if args.method in ['lara-e-gcn', 'lara-e-gat']:
        row, col = edge_index
        dgl_graph = dgl.graph((row, col))
        dgl_graph.remove_self_loop()
    
    if args.method == 'lara-n-gcn':
        pred_model = GCN(in_channels=input_feature.shape[1], hidden_channels=args.pred_hidden, 
            num_layers=args.pred_n_layers, out_channels=1, dropout=args.pred_dropout)
    elif args.method == 'lara-e-gcn':
        pred_model = GCN(in_channels=input_feature.shape[1], hidden_channels=args.pred_hidden, 
            num_layers=args.pred_n_layers, out_channels=2 * args.pred_hidden, dropout=args.pred_dropout)
    elif args.method == 'lara-n-gat':
        pred_model = GAT(in_channels=input_feature.shape[1], hidden_channels=args.pred_hidden, 
            num_layers=args.pred_n_layers, out_channels=1, dropout=args.pred_dropout)
    else:
        pred_model = GAT(in_channels=input_feature.shape[1], hidden_channels=args.pred_hidden, 
            num_layers=args.pred_n_layers, out_channels=2 * args.pred_hidden, dropout=args.pred_dropout)
    pred_model = pred_model.to(device)

    optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr, weight_decay=0)
    pred_model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        out = pred_model(input_feature, edge_index)
        if args.method in ['lara-e-gcn', 'lara-e-gat']:   # calculate the dot product of two embeddings
            dgl_graph.ndata.update({'src_emb': out[:, :args.pred_hidden]})
            dst_emb = out[:, args.pred_hidden:]
            dgl_graph.update_all(fn.copy_u("src_emb", "m"), fn.sum("m", "src_emb_sum"))
            src_emb_sum = dgl_graph.ndata['src_emb_sum']
            out = torch.mul(dst_emb, src_emb_sum)
            out = out.sum(1).unsqueeze(1)
        loss = F.mse_loss(out[:round(0.5*(len(out)))], gt[:round(0.5*len(out))])
        loss.backward()
        optimizer.step()

    new_bias_list = out.detach().squeeze(1).cpu().numpy()
    return new_bias_list
