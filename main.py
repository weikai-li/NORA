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


argparser = argparse.ArgumentParser("NORA", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--dataset", type=str, default="Cora", choices=['Cora', 'CiteSeer', 
    'PubMed', 'ogbn-arxiv', 'P50', 'P_20_50'])
argparser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT', 
    'DrGAT', 'TIMME', 'GCN-edge', 'TIMME-edge'])
argparser.add_argument('--method', type=str, default='brute', choices=['brute',
    'gradient', 'mask', 'lara-n-gcn', 'lara-e-gcn', 'lara-n-gat', 'lara-e-gat'])
argparser.add_argument('--triplet_batch_size', type=int, default=0, help="for method 'brute' and model 'TIMME-edge'")
argparser.add_argument('--n_layers', type=int, default=0, help="0 means default")
argparser.add_argument('--hidden_size', type=int, default=0, help="0 means default")
argparser.add_argument('--k1', type=float, default=0.5, help="k1 for method 'gradient' or 'MLP'")
argparser.add_argument('--k2', type=float, default=0.5, help="k2 for method 'gradient'")
argparser.add_argument('--k3', type=float, default=1000, help="k3 for method 'gradient'")
argparser.add_argument('--self_buff', type=float, default=3.0, 
    help="The ratio of self's importance to other nodes, used for method 'gradient'")
argparser.add_argument('--decay', type=float, default=1.0, help="used for method 'gradient'")
argparser.add_argument('--backward_choice', type=str, default='other_out', choices=['out', 'other_out'],
    help="used for method 'gradient'")
argparser.add_argument('--lr', type=float, default=1e-3, help="for method 'mask' and 'gcn/gat predict'")
argparser.add_argument('--n_epochs', type=int, default=100, help="for method 'mask' and 'lara")
argparser.add_argument('--alpha', type=float, default=0.1, help="for method 'mask'")
argparser.add_argument('--pred_hidden', type=int, default=64, help="for method 'lara'")
argparser.add_argument('--pred_n_layers', type=int, default=2, help="for method 'lara'")
argparser.add_argument('--pred_dropout', type=float, default=0.5, help="for method 'lara'")
args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def brute(run_time, args, device, config, graph_ori):
    if args.model == 'GCN-edge':
        if args.dataset == 'ogbn-arxiv':
            graph_ori, test_link_ori, neg_link_ori = graph_ori
        else:
            graph_ori, neg_link_ori = graph_ori
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
                ori_out_pos = ori_out[test_link_ori[0]] * ori_out[test_link_ori[1]]
                ori_out_pos = ori_out_pos.sum(1)
                ori_out_pos = torch.sigmoid(ori_out_pos)
                ori_out_neg = ori_out[neg_link_ori[0]] * ori_out[neg_link_ori[1]]
                ori_out_neg = ori_out_neg.sum(1)
                ori_out_neg = torch.sigmoid(ori_out_neg)
        num_node = graph_ori.num_nodes()
    
    elif args.dataset in ['P50', 'P_20_50']:
        features_ori, adjs_ori, triplets_ori = graph_ori
        with torch.no_grad():
            if args.model == 'TIMME':
                ori_out = model(features_ori, adjs_ori, only_classify=True)
                ori_out = torch.exp(ori_out)
            elif args.model == 'TIMME-edge':
                ori_emb = model(features_ori, adjs_ori)
                ori_out = model.calc_score_by_relation_2(triplets_ori, ori_emb[:-1], cuda=True)
        num_node = len(features_ori)

    else:    # plantoid dataset
        with torch.no_grad():
            if args.model == 'GCN-edge':
                ori_out = model(graph_ori.x, graph_ori.edge_index)
                ori_out_pos = ori_out[graph_ori.edge_index[0]] * ori_out[graph_ori.edge_index[1]]
                ori_out_pos = ori_out_pos.sum(1)
                ori_out_pos = torch.sigmoid(ori_out_pos)
                ori_out_neg = ori_out[neg_link_ori[0]] * ori_out[neg_link_ori[1]]
                ori_out_neg = ori_out_neg.sum(1)
                ori_out_neg = torch.sigmoid(ori_out_neg)
            else:
                ori_out = model(graph_ori.x, graph_ori.edge_index)
                ori_out = F.softmax(ori_out, dim=1)
        num_node = graph_ori.x.shape[0]

    new_bias_list = []
    for r_node in tqdm(range(num_node)):
        if args.dataset == 'ogbn-arxiv':
            graph = graph_ori.clone()
            graph.remove_nodes(r_node)
            feat = graph.ndata['feat']
            with torch.no_grad():
                out = model(graph, feat)
                if args.model == 'DrGAT':
                    out = F.softmax(out, dim=1)
                elif args.model in ['GCN', 'GraphSAGE']:
                    out = torch.exp(out)
                else:
                    unequal_mask = (test_link_ori != r_node)
                    unequal_mask = unequal_mask[0] & unequal_mask[1]
                    selected_out_pos = ori_out_pos[unequal_mask]
                    test_link = test_link_ori[:, unequal_mask]
                    bigger_mask = (test_link > r_node)
                    test_link[bigger_mask] = test_link[bigger_mask] - 1
                    unequal_mask = (neg_link_ori != r_node)
                    unequal_mask = unequal_mask[0] & unequal_mask[1]
                    selected_out_neg = ori_out_neg[unequal_mask]
                    neg_link = neg_link_ori[:, unequal_mask]
                    bigger_mask = (neg_link > r_node)
                    neg_link[bigger_mask] = neg_link[bigger_mask] - 1

                    out_pos = out[test_link[0]] * out[test_link[1]]
                    out_pos = out_pos.sum(1)
                    out_pos = torch.sigmoid(out_pos)
                    out_neg = out[neg_link[0]] * out[neg_link[1]]
                    out_neg = out_neg.sum(1)
                    out_neg = torch.sigmoid(out_neg)
                    bias = torch.concat([selected_out_pos, selected_out_neg]) - torch.concat([out_pos, out_neg])
                    bias = bias.abs().mean().item()
                    new_bias_list.append(bias)
                    continue
        
        elif args.model in ['TIMME', 'TIMME-edge']:
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
                continue

        else:   # planetoid dataset
            x = torch.cat((graph_ori.x[:r_node], graph_ori.x[r_node+1:]))
            edge_index = graph_ori.edge_index
            unequal_mask = (edge_index != r_node)
            unequal_mask = unequal_mask[0] & unequal_mask[1]
            edge_index = edge_index[:, unequal_mask]
            bigger_mask = (edge_index > r_node)
            edge_index[bigger_mask] = edge_index[bigger_mask] - 1
            
            if args.model != 'GCN-edge':
                with torch.no_grad():
                    out = model(x, edge_index)
                    out = F.softmax(out, dim=1)
            else:
                selected_out_pos = ori_out_pos[unequal_mask]
                unequal_mask = (neg_link_ori != r_node)
                unequal_mask = unequal_mask[0] & unequal_mask[1]
                selected_out_neg = ori_out_neg[unequal_mask]
                neg_link = neg_link_ori[:, unequal_mask]
                bigger_mask = (neg_link > r_node)
                neg_link[bigger_mask] = neg_link[bigger_mask] - 1
                out = model(x, edge_index)
                out_pos = out[edge_index[0]] * out[edge_index[1]]
                out_pos = out_pos.sum(1)
                out_pos = torch.sigmoid(out_pos)
                out_neg = out[neg_link[0]] * out[neg_link[1]]
                out_neg = out_neg.sum(1)
                out_neg = torch.sigmoid(out_neg)
                out = torch.concat([out_pos, out_neg])
                bias = torch.concat([selected_out_pos, selected_out_neg]) - torch.concat([out_pos, out_neg])
                bias = bias.abs().mean().item()
                new_bias_list.append(bias)
                continue
        
        bias = (out - torch.cat((ori_out[:r_node], ori_out[r_node+1:]))).abs()
        bias = bias.mean().item()
        new_bias_list.append(bias)
    return new_bias_list


def gradient(run_time, args, device, config, graph_ori):
    if args.dataset == 'ogbn-arxiv':
        if args.model == 'GCN-edge':
            graph_ori, test_link, neg_link = graph_ori
        graph = graph_ori.clone()
        deg = graph.in_degrees().float() - 1        # -1 is to delete self-loop
        num_node = graph.num_nodes()
    elif args.dataset in ['P50', 'P_20_50']:
        features_ori, adjs_ori, triplets_ori = graph_ori
        num_node = features_ori.shape[0]
        deg = torch.zeros(num_node, device=device)
        for i in range(10):      # Skip self-loop without counting the 10th
            new_deg = degree(adjs_ori[i].coalesce().indices()[0], num_nodes=num_node)
            deg = deg + new_deg
    else:
        if args.model == 'GCN-edge':
            graph_ori, neg_link = graph_ori
        deg = degree(graph_ori.edge_index[0])   # There is already no self-loop.
        num_node = graph_ori.x.shape[0]
    mean_deg = deg.float().mean()

    save_name = f'./save_grad/{args.dataset}_{args.model}_{args.method}'
    if args.model not in ['TIMME', 'TIMME-edge']:
        save_name += f'_{args.n_layers}_{args.hidden_size}'
    try:
        exit()
        hidden_list, hidden_grad_list = [], []
        for i in range(args.n_layers + 1):
            hidden_list.append(torch.load(f'{save_name}/{run_time}_{i}_hid.pt'))
            hidden_grad_list.append(torch.load(f'{save_name}/{run_time}_{i}_grad.pt'))
        print('load gradient information from:', save_name)
    
    except:
        model = load_model(run_time, args, device, config, graph_ori)
        model.train()
        if args.dataset == 'ogbn-arxiv':
            x = copy.deepcopy(graph.ndata['feat'])
            x.requires_grad = True
            if args.model == 'DrGAT':
                out, hidden_list = model(graph, x, return_hidden=True)
                out = F.softmax(out, dim=1)
            elif args.model in ['GCN', 'GraphSAGE']:
                out, hidden_list = model(graph, x, return_hidden=True)
                out = torch.exp(out)
            else:
                out, hidden_list = model(graph, x, return_hidden=True)
                out_pos = out[test_link[0]] * out[test_link[1]]
                out_pos = out_pos.sum(1)
                out_pos = torch.sigmoid(out_pos)
                out_neg = out[neg_link[0]] * out[neg_link[1]]
                out_neg = out_neg.sum(1)
                out_neg = torch.sigmoid(out_neg)
        elif args.dataset in ['P50', 'P_20_50']:
            x = copy.deepcopy(features_ori).to_dense()
            x.requires_grad = True
            if args.model == 'TIMME':
                out, hidden_list = model(x, adjs_ori, only_classify=True, return_hidden=True)
                out = torch.exp(out)
            elif args.model == 'TIMME-edge':
                emb, hidden_list = model(x, adjs_ori, return_hidden=True)
                out = model.calc_score_by_relation_2(triplets_ori, emb[:-1], cuda=True)
        else:
            x = copy.deepcopy(graph_ori.x)
            x.requires_grad = True
            out, hidden_list = model(x, graph_ori.edge_index, return_hidden=True)
            if args.model != 'GCN-edge':
                out = F.softmax(out, dim=1)
            else:
                out_pos = out[graph_ori.edge_index[0]] * out[graph_ori.edge_index[1]]
                out_pos = out_pos.sum(1)
                out_pos = torch.sigmoid(out_pos)
                out_neg = out[neg_link[0]] * out[neg_link[1]]
                out_neg = out_neg.sum(1)
                out_neg = torch.sigmoid(out_neg)
        
        if args.model == 'TIMME-edge':
            other_out = []
            for i in range(5):
                triplet = torch.tensor(triplets_ori[i])
                link_info1 = torch.concat([triplet[0], triplet[2]])
                link_info2 = torch.concat([triplet[2], triplet[0]])
                tmp_g = dgl.graph((link_info1, link_info2), num_nodes=num_node)
                tmp_g.edata['score'] = torch.concat([torch.tensor(out), torch.tensor(out)])
                tmp_g.remove_self_loop()
                tmp_g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'score_sum'))
                me_out = tmp_g.ndata['score_sum']
                other_out.append(out.sum() * 2 - me_out)
            other_out = torch.stack(other_out)
            print(other_out.shape)
            other_out = other_out.sum(0)
            
        elif args.model == 'GCN-edge':
            if args.dataset == 'ogbn-arxiv':
                link_info1 = torch.concat([test_link[0], test_link[1]])
                link_info2 = torch.concat([test_link[1], test_link[0]])
                tmp_g = dgl.graph((link_info1, link_info2), num_nodes=num_node)
                tmp_g.edata['score'] = torch.concat([out_pos, out_pos])
            else:
                link_info1 = torch.concat([graph_ori.edge_index[0], graph_ori.edge_index[1]])
                link_info2 = torch.concat([graph_ori.edge_index[1], graph_ori.edge_index[0]])
                tmp_g = dgl.graph((graph_ori.edge_index[0], graph_ori.edge_index[1]), num_nodes=num_node)
                tmp_g.edata['score'] = out_pos
            tmp_g.remove_self_loop()
            tmp_g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'score_sum'))
            me_out_pos = tmp_g.ndata['score_sum']

            link_info1 = torch.concat([neg_link[0], neg_link[1]])
            link_info2 = torch.concat([neg_link[1], neg_link[0]])
            tmp_g = dgl.graph((link_info1, link_info2), num_nodes=num_node)
            tmp_g = tmp_g
            tmp_g.edata['score'] = torch.concat([out_neg, out_neg])
            tmp_g.remove_self_loop()
            tmp_g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'score_sum'))
            me_out_neg = tmp_g.ndata['score_sum']
            other_out = out_pos.sum() + out_neg.sum() - me_out_pos - me_out_neg
        else:
            total_out = out.sum(0)
            other_out = total_out - out

        # x.retain_grad()
        for hs in hidden_list:
            hs.retain_grad()
        if args.backward_choice == 'out':
            out.backward(gradient=other_out, retain_graph=True)
        else:
            other_out.backward(gradient=other_out, retain_graph=True)
        os.makedirs(save_name, exist_ok=True)
        for i in range(len(hidden_list)):
            torch.save(hidden_list[i].grad.detach(), f'{save_name}/{run_time}_{i}_grad.pt')
            torch.save(hidden_list[i].detach(), f'{save_name}/{run_time}_{i}_hid.pt')
        # assert len(hidden_list) == args.n_layers + 1
        print('save gradient information at:', save_name)
        hidden_grad_list = None
    
    gradient = torch.zeros(num_node, device=device)
    rate = 1.0
    for i in range(len(hidden_list) - 2, -1, -1):
        if hidden_grad_list == None:
            new_grad = hidden_list[i].grad * hidden_list[i].detach()
        else:
            new_grad = hidden_grad_list[i] * hidden_list[i]
        new_grad = torch.norm(new_grad, p=2, dim=1)
        new_grad = new_grad * deg / (deg + args.self_buff)
        gradient = gradient + new_grad * rate
        rate = rate * (1 - mean_deg / (num_node - 1) / (mean_deg + args.self_buff))
        rate = rate * args.decay

    # new_grad = x.grad * x.detach()
    # new_grad = torch.norm(new_grad, p=2, dim=1)
    # new_grad = new_grad * deg / (deg + args.self_buff)
    # gradient = gradient + new_grad * rate

    gradient = gradient.abs()
    deg_delta1 = 1 / torch.sqrt(deg - 1) - 1 / torch.sqrt(deg)
    deg_delta2 = 1 / (deg-1) - 1 / deg
    deg_delta1[deg_delta1 == np.inf] = 1.0
    deg_delta2[deg_delta2 == np.inf] = 1.0
    assert (deg_delta1 == np.inf).sum() == 0
    assert (deg_delta2 == np.inf).sum() == 0
    deg_delta = args.k1 * deg_delta1 + (1 - args.k1) * deg_delta2
    deg_inv = args.k2 / torch.sqrt(deg) + (1 - args.k2) / deg
    
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        if args.dataset == 'Cora':
            fresh_dataset = dgl.data.CoraGraphDataset(raw_dir='./planetoid/data_dgl', verbose=False)
        elif args.dataset == 'CiteSeer':
            fresh_dataset = dgl.data.CiteseerGraphDataset(raw_dir='./planetoid/data_dgl', verbose=False)
        else:
            fresh_dataset = dgl.data.PubmedGraphDataset(raw_dir='./planetoid/data_dgl', verbose=False)
        graph = fresh_dataset[0].to(device)
        graph = graph.remove_self_loop()
        fresh_d = graph.in_degrees()
        assert ((fresh_d - deg) == 0).sum() == len(deg)
    
    elif args.dataset in ['P50', 'P_20_50']:
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
            
    if args.dataset in ['P50', 'P_20_50']:
        for i in range(1, 12):
            graph = graph.remove_self_loop(('user', f'r{i}', 'user'))
    else:
        graph = graph.remove_self_loop()

    graph.ndata.update({'deg_inv': deg_inv})
    graph.update_all(fn.copy_u("deg_inv", "m"), fn.sum("m", "deg_inv_sum"))
    deg_gather = graph.ndata['deg_inv_sum']
    graph.ndata.update({'deg_delta': deg_gather * deg_delta})
    graph.update_all(fn.copy_u("deg_delta", "m"), fn.sum("m", "deg_gather"))
    deg_gather = graph.ndata['deg_gather']
    gradient = gradient + args.k3 * len(hidden_list) * deg_gather
    return list(gradient.abs().cpu())


def main():
    global args
    if args.model != 'TIMME-edge':
        graph_ori, config = load_dataset(args, device, 0)
        args = load_default_args(args, config)
    save_name = f'./save/{args.dataset}_{args.model}_{args.method}'
    if args.model not in ['TIMME', 'TIMME-edge']:
        save_name += f'_{args.n_layers}_{args.hidden_size}'
    
    bias_list = []
    start_time = time.time()
    for run_time in range(0, 5):
        if args.model == 'TIMME-edge':
            graph_ori, config = load_dataset(args, device, run_time)
            args = load_default_args(args, config)
        if args.method == 'gradient':
            new_bias_list = gradient(run_time, args, device, config, graph_ori)
        elif args.method == 'brute':
            new_bias_list = brute(run_time, args, device, config, graph_ori)
        elif args.method == 'mask':
            new_bias_list = node_mask(run_time, args, device, config, graph_ori)
        elif args.method in ['lara-n-gcn', 'lara-e-gcn', 'lara-n-gat', 'lara-e-gat']:
            new_bias_list = gnn_predict(run_time, args, device, config, graph_ori)
        bias_list.append(new_bias_list)
        if args.method == 'brute':
            np.save(f'{save_name}_mean.npy', bias_list)
            print('saving:', f'{save_name}_mean.npy')
    time_cost = time.time() - start_time
    print(args.method, args.dataset, args.model, 'time cost:', time_cost)

    if args.method != 'brute':
        np.save(f'{save_name}.npy', bias_list)
        print('saving:', f'{save_name}_mean.npy')


if __name__ == '__main__':
    main()
