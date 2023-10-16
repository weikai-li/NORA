import numpy as np
import torch
import argparse
import dgl
import scipy.stats as stats
import matplotlib.pyplot as plt
from planetoid.best_config import config as config_ori
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx as pyg_to_networkx
import networkx as nx
from load import load_dataset, load_default_args
import time


argparser = argparse.ArgumentParser("Bias Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--dataset", type=str, default="Cora", choices=['Cora', 
    'CiteSeer', 'PubMed', 'ogbn-arxiv', 'P50', 'P_20_50'])
argparser.add_argument('--model', type=str, default='GCN', choices=['GCN', 
    'GraphSAGE', 'GAT', 'DrGAT', 'TIMME', 'GCN-edge', 'TIMME-edge'])
argparser.add_argument('--analyze', type=str, default='performance', choices=['performance', 'stability'],
    help="performance: basic analysis; \
    stability: analyze stability of different models and hyper-parameters")
argparser.add_argument('--n_layers', type=int, default=0)
argparser.add_argument('--hidden_size', type=int, default=0, help="0 means default")
argparser.add_argument('--method', type=str, default='gradient', choices=['gradient', 'degree',
    'betweenness', 'mask', 'lara-n-gcn', 'lara-e-gcn', 'lara-n-gat', 'lara-e-gat'])
args = argparser.parse_args()


if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    config = config_ori[args.model][args.dataset]
else:
    config = None
args = load_default_args(args, config)

    
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
        if args.method in ['lara-n', 'lara-e']:
            res = res[round(0.5 * len(res)):]
        return res.mean(0)
    
    elif method == 'degree':
        graph_ori, config = load_dataset(args, device='cpu')
        if args.dataset == 'ogbn-arxiv':
            deg = graph_ori.in_degrees().float() - 1
        elif args.dataset in ['P50', 'P_20_50']:
            features_ori, adjs_ori, link_info = graph_ori
            num_node = features_ori.shape[0]
            deg = torch.zeros(num_node)
            for i in range(10):      # Skip self-loop without counting the 10th
                new_deg = degree(adjs_ori[i].coalesce().indices()[0], num_nodes=num_node)
                deg = deg + new_deg
        else:
            deg = degree(graph_ori.edge_index[0])   # There is already no self-loop.
        return deg
    
    else:    # betweenness
        graph_ori, config = load_dataset(args, device='cpu')
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            try:
                exit()
                betweenness = np.load(f'./planetoid/data/{args.dataset}/processed/betweenness.npy')
            except:
                print('generating betweenness centrality')
                num_node = graph_ori.x.shape[0]
                data = pyg_to_networkx(graph_ori, to_undirected=True)
                start_time = time.time()
                betweenness = nx.betweenness_centrality(data)
                end_time = time.time()
                print(end_time - start_time)
                betweenness = np.array([betweenness[i] for i in range(num_node)])
                np.save(f'./planetoid/data/{args.dataset}/processed/betweenness.npy', betweenness)
        elif args.dataset == 'ogbn-arxiv':
            try:
                betweenness = np.load('./arxiv/dataset/ogbn_arxiv/processed/betweenness.npy')
            except:
                print('generating betweenness centrality')
                num_node = graph_ori.num_nodes()
                data = dgl.to_networkx(graph_ori)
                betweenness = nx.betweenness_centrality(data, k=10000)
                betweenness = np.array([betweenness[i] for i in range(num_node)])
                np.save('./arxiv/dataset/ogbn_arxiv/processed/betweenness.npy', betweenness)
        else:
            try:
                betweenness = np.load(f'./TIMME/data/{args.dataset}/betweenness.npy')
            except:
                print('generating betweenness centrality')
                features, adjs, link_info = graph_ori
                num_node = adjs[0].shape[0]
                edge0 = [adjs[i]._indices()[0] for i in range(10)]
                edge1 = [adjs[i]._indices()[1] for i in range(10)]
                edge0 = torch.concat(edge0)
                edge1 = torch.concat(edge1)
                graph = dgl.graph((edge0, edge1))
                data = dgl.to_networkx(graph)
                betweenness = nx.betweenness_centrality(data)
                betweenness = np.array([betweenness[i] for i in range(num_node)])
                np.save(f'./TIMME/data/{args.dataset}/betweenness.npy', betweenness)
        return betweenness


def performance():
    gt = load_results('brute', args)
    res = load_results(args.method, args)
    gt_sort = np.argsort(gt)
    res_sort = np.argsort(res)
    
    # assert (res<0).sum() == 0
    perform_list = [gt[res_sort[-1]] / gt[gt_sort[-1]] * 100]

    for rate in [0.05, 0.10]:
        res_perform = gt[res_sort[-round(rate * len(gt)):]].sum()
        gt_perform = gt[gt_sort[-round(rate * len(gt)):]].sum()
        perform_list.append(res_perform / gt_perform * 100)

    output = ''
    for i in perform_list:
        output += f'{i:.1f}\% & '
    if args.method == 'gradient':
        output += f'\\textbf{stats.pearsonr(gt, res)[0]:.3f} & '
    else:
        output += f'{stats.pearsonr(gt, res)[0]:.3f} & '
    print(output)


def stability_arxiv():
    list1 = np.load(f'./save/ogbn-arxiv_GCN_brute_3_128_mean.npy', allow_pickle=True).mean(0)
    list2 = np.load(f'./save/ogbn-arxiv_GCN_brute_3_mean.npy', allow_pickle=True).mean(0)
    list3 = np.load(f'./save/ogbn-arxiv_GCN_brute_3_512_mean.npy', allow_pickle=True).mean(0)
    list4 = np.load(f'./save/ogbn-arxiv_GraphSAGE_brute_3_128_mean.npy', allow_pickle=True).mean(0)
    list5 = np.load(f'./save/ogbn-arxiv_GraphSAGE_brute_3_mean.npy', allow_pickle=True).mean(0)
    list6 = np.load(f'./save/ogbn-arxiv_GraphSAGE_brute_3_512_mean.npy', allow_pickle=True).mean(0)
    list7 = np.load(f'./save/ogbn-arxiv_DrGAT_brute_3_128_mean.npy', allow_pickle=True).mean(0)
    list8 = np.load(f'./save/ogbn-arxiv_DrGAT_brute_3_mean.npy', allow_pickle=True).mean(0)
    assert list2.shape[0] > 1000

    pearson_list = [
        np.mean([stats.pearsonr(list1, list2)[0], stats.pearsonr(list2, list3)[0], stats.pearsonr(list1, list3)[0]]),
        np.mean([stats.pearsonr(list4, list5)[0], stats.pearsonr(list5, list6)[0], stats.pearsonr(list4, list6)[0]]),
        stats.pearsonr(list7, list8)[0]
    ]
    print('Pearson correlation coefficient of different hidden sizes:')
    print(pearson_list)
    # print(np.mean(pearson_list), np.std(pearson_list))
    print()

    print('Pearson correlation coefficient of different GNNs:')
    pearson_list = [
        stats.pearsonr(list1, list4)[0], stats.pearsonr(list2, list5)[0], stats.pearsonr(list3, list6)[0],
        stats.pearsonr(list4, list7)[0], stats.pearsonr(list5, list8)[0],
        stats.pearsonr(list1, list7)[0], stats.pearsonr(list2, list8)[0]
    ]
    # print(pearson_list)
    print(np.mean(pearson_list), np.std(pearson_list))


def stability_planetoid():
    if args.dataset in ['Cora', 'PubMed']:
        list1 = np.load(f'./save/{args.dataset}_GCN_brute_2_128_mean.npy', allow_pickle=True).mean(0)
        list2 = np.load(f'./save/{args.dataset}_GCN_brute_2_256_mean.npy', allow_pickle=True).mean(0)
        list3 = np.load(f'./save/{args.dataset}_GCN_brute_2_512_mean.npy', allow_pickle=True).mean(0)
        list4 = np.load(f'./save/{args.dataset}_GraphSAGE_brute_2_128_mean.npy', allow_pickle=True).mean(0)
        list5 = np.load(f'./save/{args.dataset}_GraphSAGE_brute_2_mean.npy', allow_pickle=True).mean(0)
        list6 = np.load(f'./save/{args.dataset}_GraphSAGE_brute_2_512_mean.npy', allow_pickle=True).mean(0)
        list7 = np.load(f'./save/{args.dataset}_GAT_brute_2_128_mean.npy', allow_pickle=True).mean(0)
        list8 = np.load(f'./save/{args.dataset}_GAT_brute_2_256_mean.npy', allow_pickle=True).mean(0)
        list9 = np.load(f'./save/{args.dataset}_GAT_brute_2_512_mean.npy', allow_pickle=True).mean(0)
    elif args.dataset == 'CiteSeer':
        list1 = np.load(f'./save/{args.dataset}_GCN_brute_2_128_mean.npy', allow_pickle=True).mean(0)
        list2 = np.load(f'./save/{args.dataset}_GCN_brute_2_256_mean.npy', allow_pickle=True).mean(0)
        list3 = np.load(f'./save/{args.dataset}_GCN_brute_2_512_mean.npy', allow_pickle=True).mean(0)
        list4 = np.load(f'./save/{args.dataset}_GraphSAGE_brute_2_mean.npy', allow_pickle=True).mean(0)
        list5 = np.load(f'./save/{args.dataset}_GraphSAGE_brute_2_256_mean.npy', allow_pickle=True).mean(0)
        list6 = np.load(f'./save/{args.dataset}_GraphSAGE_brute_2_512_mean.npy', allow_pickle=True).mean(0)
        list7 = np.load(f'./save/{args.dataset}_GAT_brute_2_128_mean.npy', allow_pickle=True).mean(0)
        list8 = np.load(f'./save/{args.dataset}_GAT_brute_2_mean.npy', allow_pickle=True).mean(0)
        list9 = np.load(f'./save/{args.dataset}_GAT_brute_2_512_mean.npy', allow_pickle=True).mean(0)
    assert list2.shape[0] > 1000

    pearson_list = [
        np.mean([stats.pearsonr(list1, list2)[0], stats.pearsonr(list2, list3)[0], stats.pearsonr(list1, list3)[0]]),
        np.mean([stats.pearsonr(list4, list5)[0], stats.pearsonr(list5, list6)[0], stats.pearsonr(list4, list6)[0]]),
        np.mean([stats.pearsonr(list7, list8)[0], stats.pearsonr(list8, list9)[0], stats.pearsonr(list7, list9)[0]])
    ]
    print('Pearson correlation coefficient of different hidden sizes:')
    print(pearson_list)
    # print(np.mean(pearson_list), np.std(pearson_list))
    print()

    print('Pearson correlation coefficient of different GNNs:')
    pearson_list = [
        stats.pearsonr(list1, list4)[0], stats.pearsonr(list2, list5)[0], stats.pearsonr(list3, list6)[0],
        stats.pearsonr(list4, list7)[0], stats.pearsonr(list5, list8)[0], stats.pearsonr(list6, list9)[0],
        stats.pearsonr(list1, list7)[0], stats.pearsonr(list2, list8)[0], stats.pearsonr(list3, list9)[0]
    ]
    print(pearson_list)
    print(np.mean(pearson_list), np.std(pearson_list))


if __name__ == '__main__':
    if args.analyze == 'performance':
        performance()
    elif args.analyze == 'stability':
        if args.dataset == 'ogbn-arxiv':
            stability_arxiv()
        elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            stability_planetoid()
