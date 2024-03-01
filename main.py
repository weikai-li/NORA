import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import argparse
from tqdm import tqdm
import copy
import time
import dgl
import dgl.function as fn
import os
from load import load_model, load_dataset, load_default_args
from baseline import node_mask, gnn_predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def brute(run_time, args, device, graph_ori):
    if args.model[-4:] == 'edge' and args.dataset not in ['P50', 'P_20_50']:
        graph_ori, pred_edge_ori = graph_ori
    model = load_model(run_time, args, device, graph_ori)
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        graph_ori = graph_ori[0].to(device)
        deg = graph_ori.in_degrees()
    model.eval()

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
            num_node = len(features_ori)

    new_bias_list = []
    for r_node in tqdm(range(num_node)):
        if args.dataset in ['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed']:
            graph = graph_ori.clone()
            graph.remove_nodes(r_node)
            with torch.no_grad():
                out = model(graph, graph.ndata['feat'])
                if args.model[-4:] != 'edge':
                    out = F.softmax(out, dim=1)
                    bias = (out - torch.cat((ori_out[:r_node], ori_out[r_node+1:]))).abs()
                else:
                    unequal_mask = (pred_edge_ori != r_node)
                    unequal_mask = unequal_mask[0] & unequal_mask[1]
                    selected_out = ori_out[unequal_mask]
                    pred_edge = pred_edge_ori[:, unequal_mask]
                    bigger_mask = (pred_edge > r_node)
                    pred_edge[bigger_mask] = pred_edge[bigger_mask] - 1
                    out = out[pred_edge[0]] * out[pred_edge[1]]
                    out = out.sum(1)
                    out = torch.sigmoid(out)
                    bias = (out - selected_out).abs()
                
        elif args.model in ['TIMME', 'TIMME_edge']:
            feat_indices = features_ori._indices()
            feat_indices1 = torch.cat((feat_indices[0, :r_node], feat_indices[0, r_node + 1:] - 1))
            feat_indices2 = torch.cat((feat_indices[1, :r_node], feat_indices[1, r_node + 1:]))
            feat_indices = torch.stack([feat_indices1, feat_indices2])
            feat_values = features_ori._values()[1:]
            features = torch.sparse_coo_tensor(indices=feat_indices, values=feat_values, size=[num_node-1, num_node])

            adjs = []
            for adj_ori in adjs_ori:
                values = adj_ori._values()
                indices = adj_ori._indices()
                unequal_mask = (indices != r_node)
                unequal_mask = unequal_mask[0] & unequal_mask[1]
                indices = indices[:, unequal_mask]
                values = values[unequal_mask]
                bigger_mask = (indices > r_node)
                indices[bigger_mask] = indices[bigger_mask] - 1
                adj = torch.sparse_coo_tensor(indices=indices, values=values, size=[num_node - 1, num_node - 1])
                adjs.append(adj)
            
            if args.model == 'TIMME':
                with torch.no_grad():
                    out = model(features, adjs, only_classify=True)
                    out = torch.exp(out)
            else:
                triplets = []
                ori_out_selected = []
                for tri_i, triplet in enumerate(triplets_ori):
                    unequal_mask = (triplet[0] != r_node) & (triplet[2] != r_node)
                    triplet = triplet[:, unequal_mask]
                    ori_out_selected.append(ori_out[tri_i][unequal_mask])
                    bigger_mask0 = triplet[0] > r_node
                    triplet[0, bigger_mask0] = triplet[0, bigger_mask0] - 1
                    bigger_mask2 = triplet[2] > r_node
                    triplet[2, bigger_mask2] = triplet[2, bigger_mask2] - 1
                    triplets.append(triplet)
                with torch.no_grad():
                    emb = model(features, adjs)
                    out = model.calc_score_by_relation_2(triplets, emb[:-1], cuda=True)
                bias = (torch.cat(out) - torch.cat(ori_out_selected)).abs()
        
        bias = bias.mean().item()
        new_bias_list.append(bias)
    return new_bias_list


def nora(run_time, args, device, graph_ori):
    if args.model[-4:] == 'edge' and args.dataset not in ['P50', 'P_20_50']:
        graph_ori, pred_edge_ori = graph_ori
    model = load_model(run_time, args, device, graph_ori)
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        graph_ori = graph_ori[0].to(device)
    if args.dataset in ['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed']:
        graph = graph_ori.clone()
        deg = graph.remove_self_loop().in_degrees().float()
        num_node = graph.num_nodes()
    else:
        assert args.dataset in ['P50', 'P_20_50']
        features_ori, adjs_ori, triplets_ori = graph_ori
        num_node = features_ori.shape[0]
        deg = torch.zeros(num_node, device=device)
        for i in range(10):      # Skip self-loop without counting the 10th
            new_deg = degree(adjs_ori[i].coalesce().indices()[0], num_nodes=num_node)
            deg = deg + new_deg
    mean_deg = deg.float().mean()
    if args.model == 'DrGAT':
        model.train()    # DrGAT does not support keeping gradient with model.eval()
    else:
        model.eval()   # Using eval() can make the model more robust
    
    if args.dataset in ['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed']:
        x = copy.deepcopy(graph.ndata['feat'])
        x.requires_grad = True
        out, hidden_list = model(graph, x, return_hidden=True)
        if args.model[-4:] != 'edge':
            out = F.softmax(out, dim=1)
        else:
            out = out[pred_edge_ori[0]] * out[pred_edge_ori[1]]
            out = out.sum(1)
            out = torch.sigmoid(out)
    elif args.dataset in ['P50', 'P_20_50']:
        x = copy.deepcopy(features_ori).to_dense()
        x.requires_grad = True
        if args.model == 'TIMME':
            out, hidden_list = model(x, adjs_ori, only_classify=True, return_hidden=True)
            out = torch.exp(out)
        elif args.model == 'TIMME_edge':
            emb, hidden_list = model(x, adjs_ori, return_hidden=True)
            out = model.calc_score_by_relation_2(triplets_ori, emb[:-1], cuda=True)
    
    # For link prediction outputs, we assign the output scores to nodes
    if args.model[-4:] == 'edge':
        if args.dataset in ['P50', 'P_20_50']:
            tmp_out_list = []
            for i in range(len(triplets_ori)):
                triplet = torch.tensor(triplets_ori[i])
                link_info1 = torch.concat([triplet[0], triplet[2]])
                link_info2 = torch.concat([triplet[2], triplet[0]])
                tmp_g = dgl.graph((link_info1, link_info2), num_nodes=num_node).to(device)
                tmp_g.edata['score'] = torch.concat([out[i], out[i]])
                tmp_g.update_all(fn.copy_e('score', 'score_m'), fn.sum('score_m', 'score_sum'))
                tmp_out = tmp_g.ndata['score_sum']
                tmp_out_list.append(tmp_out)
            out = torch.stack(tmp_out_list).mean(0)
        else:
            link_info1 = torch.concat([pred_edge_ori[0], pred_edge_ori[1]])
            link_info2 = torch.concat([pred_edge_ori[1], pred_edge_ori[0]])
            tmp_g = dgl.graph((link_info1, link_info2), num_nodes=num_node).to(device)
            tmp_g.edata['score'] = torch.concat([out, out])
            tmp_g.update_all(fn.copy_e('score', 'score_m'), fn.sum('score_m', 'score_sum'))
            out = tmp_g.ndata['score_sum']

    for hs in hidden_list:
        hs.retain_grad()
    out.backward(gradient=out, retain_graph=True)
    hidden_grad_list = []
    for i in range(len(hidden_list)):
        hidden_grad_list.append(hidden_list[i].grad.detach())

    gradient = torch.zeros(num_node, device=device)
    rate = 1.0
    assert len(hidden_list) == args.num_layers + 1
    for i in range(len(hidden_list) - 2, -1, -1):
        new_grad = hidden_grad_list[i] * hidden_list[i]
        new_grad = torch.norm(new_grad, p=args.grad_norm, dim=1)
        new_grad = new_grad * deg / (deg + args.self_buff)
        gradient = gradient + new_grad * rate
        # grad_sum = hidden_grad_list[i].abs().sum()
        # grad_sum_ratio = hidden_grad_list[i].abs().sum(1) / grad_sum
        # rate = rate * (1 - grad_sum_ratio * deg / (deg + args.self_buff))
        rate = rate * (1 - deg / (num_node - 1) / (mean_deg + args.self_buff))

    assert (gradient < 0).sum() == 0
    deg_delta1 = 1 / torch.sqrt(deg - 1) - 1 / torch.sqrt(deg)
    deg_delta2 = 1 / (deg-1) - 1 / deg
    deg_delta1[deg_delta1 == np.nan] = 1.0
    deg_delta2[deg_delta2 == np.nan] = 1.0
    deg_delta1[deg_delta1.abs() == np.inf] = 1.0
    deg_delta2[deg_delta2.abs() == np.inf] = 1.0
    deg_delta = args.k1 * deg_delta1 + (1 - args.k1) * deg_delta2
    deg_inv = args.k2[0] / torch.sqrt(deg) + args.k2[1] / deg + (1 - args.k2[0] - args.k2[1])
    
    if args.dataset in ['P50', 'P_20_50']:
        num_nodes_dict = {'user': features_ori.shape[0]}
        adjs = adjs_ori
        data_dict = {
            ('user', 'r1', 'user'): (adjs[0]._indices()[0], adjs[0]._indices()[1]),
            ('user', 'r2', 'user'): (adjs[1]._indices()[0], adjs[1]._indices()[1]),
            ('user', 'r3', 'user'): (adjs[2]._indices()[0], adjs[2]._indices()[1]),
            ('user', 'r4', 'user'): (adjs[3]._indices()[0], adjs[3]._indices()[1]),
            ('user', 'r5', 'user'): (adjs[4]._indices()[0], adjs[4]._indices()[1]),
            ('user', 'r6', 'user'): (adjs[5]._indices()[0], adjs[5]._indices()[1]),
            ('user', 'r7', 'user'): (adjs[6]._indices()[0], adjs[6]._indices()[1]),
            ('user', 'r8', 'user'): (adjs[7]._indices()[0], adjs[7]._indices()[1]),
            ('user', 'r9', 'user'): (adjs[8]._indices()[0], adjs[8]._indices()[1]),
            ('user', 'r10', 'user'): (adjs[9]._indices()[0], adjs[9]._indices()[1]),
            ('user', 'r11', 'user'): (adjs[10]._indices()[0], adjs[10]._indices()[1]),
        }
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        for i in range(1, 12):
            graph = graph.remove_self_loop(('user', f'r{i}', 'user'))
    else:
        graph = graph.remove_self_loop()

    graph.ndata.update({'deg_inv': deg_inv})
    graph.update_all(fn.copy_u("deg_inv", "m1"), fn.sum("m1", "deg_inv_sum"))
    deg_gather = graph.ndata['deg_inv_sum']
    graph.ndata.update({'deg_delta': deg_gather * deg_delta})
    graph.update_all(fn.copy_u("deg_delta", "m2"), fn.sum("m2", "deg_gather"))
    deg_gather = graph.ndata['deg_gather']
    deg_gather = deg_gather / deg_gather.mean() * gradient.mean()  # Normalize
    gradient = gradient + args.k3 * deg_gather
    gradient = gradient.abs().detach().cpu().numpy()
    return gradient


def main():
    argparser = argparse.ArgumentParser("NORA", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--dataset", type=str, default="Cora", choices=['Cora', 'CiteSeer', 
        'PubMed', 'ogbn-arxiv', 'P50', 'P_20_50'])
    argparser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT', 
        'DrGAT', 'GCNII', 'TIMME', 'GCN_edge', 'GraphSAGE_edge', 'GAT_edge', 'GCNII_edge', 'TIMME_edge'])
    argparser.add_argument('--method', type=str, default='brute', choices=['brute',
        'nora', 'mask', 'gcn-n', 'gcn-e', 'gat-n', 'gat-e'])
    argparser.add_argument('--cycles', type=int, nargs='+', default=[], help="[] means all")
    # Hyper-parameters for loading the original trained model
    argparser.add_argument('--triplet_batch_size', type=int, default=0, help="for model 'TIMME_edge'")
    argparser.add_argument('--num_layers', type=int, default=0, help="0 means default")
    argparser.add_argument('--hidden_size', type=int, default=0, help="0 means default")
    argparser.add_argument('--dropout', type=int, default=0, help="0 means default")
    # Hyper-parameters for the approximation methods
    argparser.add_argument('--k1', type=float, default=0.5, help="k1 for method 'nora' or 'MLP'")
    argparser.add_argument('--k2', type=float, default=[0.5, 0.5], nargs='+', help="k2 for method 'nora'")
    argparser.add_argument('--k3', type=float, default=1000, help="k3 for method 'nora'")
    argparser.add_argument('--self_buff', type=float, default=3.0, 
        help="The ratio of self's importance to other nodes, used for method 'nora'")
    argparser.add_argument('--grad_norm', type=float, default=1, help="used for method 'nora'")
    argparser.add_argument('--lr', type=float, default=1e-3, help="for method 'mask' and 'gcn/gat predict'")
    argparser.add_argument('--wd', type=float, default=0, help="for method 'mask' and 'gcn/gat predict'")
    argparser.add_argument('--num_epochs', type=int, default=100, help="for method 'mask' and 'gcn/gat")
    argparser.add_argument('--alpha', type=float, default=0.1, help="for method 'mask'")
    argparser.add_argument('--pred_hidden', type=int, default=64, help="for method 'gcn/gat'")
    argparser.add_argument('--pred_num_layers', type=int, default=2, help="for method 'gcn/gat'")
    argparser.add_argument('--pred_dropout', type=float, default=0.5, help="for method 'gcn/gat'")
    argparser.add_argument('--train_ratio', type=float, default=0, help="for method 'gcn/gat'")
    argparser.add_argument('--val_ratio', type=float, default=0, help="for method 'gcn/gat'")
    args = argparser.parse_args()
    args = load_default_args(args)
    print(args)

    os.makedirs('save', exist_ok=True)
    bias_list = []
    start_time = time.time()
    if args.cycles == []:
        args.cycles = [0, 1, 2, 3, 4]
    for run_time in args.cycles:
        graph_ori = load_dataset(args, device, run_time)
        if args.method == 'nora':
            new_bias_list = nora(run_time, args, device, graph_ori)
        elif args.method == 'brute':
            new_bias_list = brute(run_time, args, device, graph_ori)
        elif args.method == 'mask':
            new_bias_list = node_mask(run_time, args, device, graph_ori)
        elif args.method in ['gcn-n', 'gcn-e', 'gat-n', 'gat-e']:
            new_bias_list = gnn_predict(run_time, args, device, graph_ori)
        bias_list.append(new_bias_list)
        if args.method == 'brute':
            save_name = f'./save/{args.dataset}_{args.model}_{args.method}'
            if args.model not in ['TIMME', 'TIMME_edge']:
                save_name += f'_{args.num_layers}_{args.hidden_size}'
            if args.cycles != [0, 1, 2, 3, 4]:
                save_name += f'_{run_time}'
            np.save(f'{save_name}.npy', bias_list)
            print('saving:', f'{save_name}.npy')
    time_cost = time.time() - start_time
    print(args.method, args.dataset, args.model, 'time cost:', time_cost)

    save_name = f'./save/{args.dataset}_{args.model}_{args.method}'
    if args.model not in ['TIMME', 'TIMME_edge']:
        save_name += f'_{args.num_layers}_{args.hidden_size}'
    if args.method != 'brute':
        np.save(f'{save_name}.npy', bias_list)
        print('saving:', f'{save_name}.npy')


if __name__ == '__main__':
    main()
