import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import dgl
import dgl.function as fn
import copy
from load import load_model, load_results
from planetoid.models import MyGCN, MyGAT


def node_mask(run_time, args, device, graph_ori):
    if args.model[-4:] == 'edge' and args.dataset not in ['P50', 'P_20_50']:
        graph_ori, pred_edge_ori = graph_ori
    model = load_model(run_time, args, device, graph_ori)
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        graph_ori = graph_ori[0].to(device)
    
    model.eval()    # Using model.eval() can make the outputs more stable
    with torch.no_grad():
        if args.dataset in ['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed']:
            ori_out = model(graph_ori, graph_ori.ndata['feat'])
            if args.model[-4:] != 'edge':
                ori_out = F.softmax(ori_out, dim=1)
            else:
                ori_out = ori_out[pred_edge_ori[0]] * ori_out[pred_edge_ori[1]]
                ori_out = ori_out.sum(1)
                ori_out = torch.sigmoid(ori_out)
            num_node = graph_ori.num_nodes()
        elif args.dataset in ['P50', 'P_20_50']:
            features_ori, adjs_ori, triplets_ori = graph_ori
            if args.model == 'TIMME':
                ori_out = model(features_ori, adjs_ori, only_classify=True)
                ori_out = torch.exp(ori_out)
            elif args.model == 'TIMME_edge':
                ori_emb = model(features_ori, adjs_ori)
                ori_out = model.calc_score_by_relation_2(triplets_ori, ori_emb[:-1], cuda=True)
                ori_out = torch.concat(ori_out)
            num_node = len(features_ori)
    
    weight = torch.ones((num_node, 1), device=device)
    weight.requires_grad = True
    optimizer = torch.optim.Adam([weight], lr=args.lr, weight_decay=args.wd)
    for m in model.parameters():
        m.requires_grad = False

    num_val = round(args.val_ratio * num_node)
    best_val, best_weight = 1e5, None
    for epoch in tqdm(range(args.num_epochs)):
        optimizer.zero_grad()
        if args.dataset in ['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed']:
            out = model(graph_ori, graph_ori.ndata['feat'], node_weight=weight)
            if args.model[-4:] != 'edge':
                out = F.softmax(out, dim=1)
            else:
                out = out[pred_edge_ori[0]] * out[pred_edge_ori[1]]
                out = out.sum(1)
                out = torch.sigmoid(out)
        elif args.dataset in ['P50', 'P_20_50']:
            if args.model == 'TIMME':
                out = model(features_ori, adjs_ori, only_classify=True, node_weight=weight)
                out = torch.exp(out)
            elif args.model == 'TIMME-edge':
                emb = model(features_ori, adjs_ori, node_weight=weight)
                out = model.calc_score_by_relation_2(triplets_ori, emb[:-1], cuda=True)
                out = torch.concat(out)
        
        loss1 = - (out - ori_out).abs()
        val_loss = loss1[:num_val].mean()
        loss2 = (num_node - weight.sum()) / num_node
        loss = loss1.mean() + args.alpha * loss2
        loss.backward()
        optimizer.step()
        weight.requires_grad = False
        weight[weight < 0.0] = 0.0
        weight[weight > 1.0] = 1.0
        weight.requires_grad = True
        if val_loss < best_val:
            best_val = val_loss
            best_weight = weight.detach().clone()
    new_bias_list = (1 - best_weight).cpu().numpy().squeeze(1)
    return new_bias_list


def gnn_predict(run_time, args, device, graph):
    gt = load_results(args, 'brute')
    gt = torch.tensor(gt, dtype=torch.float32).to(device).unsqueeze(1)
    gt = (gt - gt.mean()) / gt.std() - gt.min()
    
    if args.model[-4:] == 'edge' and args.dataset not in ['P50', 'P_20_50']:
        graph, pred_edge_ori = graph
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        graph = graph[0].to(device)
    elif args.dataset in ['P50', 'P_20_50']:
        feature, adjs, triplets = graph
        edge_index = [adj.coalesce().indices() for adj in adjs]
        edge_index = torch.concat(edge_index, 1)
        graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=feature.shape[0]).to(device)
        graph.ndata['feat'] = feature
    
    if args.method == 'gcn-n':
        pred_model = MyGCN(graph.ndata["feat"].shape[1], args.pred_hidden, 1,
            args.pred_num_layers, args.pred_dropout)
    elif args.method == 'gcn-e':
        pred_model = MyGCN(graph.ndata["feat"].shape[1], args.pred_hidden, 2 * args.pred_hidden,
            args.pred_num_layers, args.pred_dropout)
    elif args.method == 'gat-n':
        pred_model = MyGAT(graph.ndata["feat"].shape[1], args.pred_hidden, 1,
            args.pred_num_layers, args.pred_dropout)
    else:
        pred_model = MyGAT(graph.ndata["feat"].shape[1], args.pred_hidden, 2 * args.pred_hidden,
            args.pred_num_layers, args.pred_dropout)
    pred_model = pred_model.to(device)

    optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val, best_out = 10e5, None
    num_train = round(args.train_ratio * graph.num_nodes())
    num_val = round(args.val_ratio * graph.num_nodes())
    for epoch in tqdm(range(args.num_epochs)):
        pred_model.train()
        optimizer.zero_grad()
        out = pred_model(graph, graph.ndata['feat'])
        if args.method in ['gcn-e', 'gat-e']:  # Calculate a score for each edge and add them
            graph.ndata.update({'src_emb': out[:, :args.pred_hidden]})
            dst_emb = out[:, args.pred_hidden:]
            graph.update_all(fn.copy_u("src_emb", "m"), fn.sum("m", "src_emb_sum"))
            src_emb_sum = graph.ndata['src_emb_sum']
            out = torch.mul(dst_emb, src_emb_sum)
            out = out.sum(1).unsqueeze(1)
        loss = F.mse_loss(out[:num_train], gt[:num_train])
        loss.backward()
        optimizer.step()
        val_loss = F.mse_loss(out[num_train: num_train+num_val], gt[num_train: num_train+num_val])
        if val_loss < best_val:
            best_val = val_loss
            best_out = out.detach()

    new_bias_list = best_out.squeeze(1).cpu().numpy()
    return new_bias_list
