import numpy as np
import torch
import torch.nn.functional as F
import json
import argparse
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import dgl
from load import load_default_args, load_results, load_dataset, load_model
from arxiv.simple.train import load_dataset as load_arxiv_dataset
from TIMME.code.utils import multi_relation_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True  # Use type 1 font
# matplotlib.rcParams['text.usetex'] = True


def performance(args):
    gt = load_results(args, 'brute')
    res = load_results(args, args.method)
    # gt_sort = np.argsort(gt)
    # res_sort = np.argsort(res)
    # assert (res<0).sum() == 0
    res[res < 0] = 0

    if args.method in ['gcn-n', 'gcn-e', 'gat-n', 'gat-e']:
        num_train = round(args.train_ratio * len(gt))
    else:
        num_train = 0
    num_val = round(args.val_ratio * len(gt))
    val_perf = stats.pearsonr(gt[num_train:num_train+num_val], res[num_train:num_train+num_val])[0]
    test_perf = stats.pearsonr(gt[num_train+num_val:], res[num_train+num_val:])[0]
    print(f'val: {val_perf:.3f}, test: {test_perf:.3f}')
    # perform_list = [gt[res_sort[-1]] / gt[gt_sort[-1]] * 100]
    # for rate in [0.05, 0.10]:
    #     res_perform = gt[res_sort[-round(rate * len(gt)):]].sum()
    #     gt_perform = gt[gt_sort[-round(rate * len(gt)):]].sum()
    #     perform_list.append(res_perform / gt_perform * 100)
    # output = ''
    # for i in perform_list:
        # output += f'{i:.1f}\% & '
    # if args.method == 'nora':
        # output += f'\\textbf{stats.pearsonr(gt, res)[0]:.3f} & '
    # else:
        # output += f'{stats.pearsonr(gt, res)[0]:.3f} & '
    # print(output)


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


def explain(args):
    stability_list, perf_list = [], []
    dataset_list = ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv']
    model_list = ['GCN', 'GraphSAGE', 'GAT', 'GCNII']
    # model_list = ['GCN_edge', 'GraphSAGE_edge', 'GAT_edge']
    # dataset_list = ['P50', 'P_20_50']
    # model_list = ['TIMME', 'TIMME_edge']

    for m in model_list:
        tmp_stable_list, tmp_perf_list = [], []
        for d in dataset_list:
            args.dataset = d
            if d == 'ogbn-arxiv' and m == 'GAT':
                args.model = 'DrGAT'
            else:
                args.model = m
            args.num_layers = args.hidden_size = 0
            args = load_default_args(args)
            gt = load_results(args, 'brute')
            res = load_results(args, args.method)
            res[res < 0] = 0
            if args.method in ['gcn-n', 'gcn-e', 'gat-n', 'gat-e']:
                num_train = round(args.train_ratio * len(gt))
            else:
                num_train = 0
            num_val = round(args.val_ratio * len(gt))
            test_perf = stats.pearsonr(gt[num_train+num_val:], res[num_train+num_val:])[0]
            stability = gt.std() / gt.mean()

            # graph_ori = load_dataset(args, device='cpu', run_time=0)
            # if args.model[-4:] == 'edge' and args.dataset not in ['P50', 'P_20_50']:
            #     graph_ori, pred_edge_ori = graph_ori
            # if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            #     graph_ori = graph_ori[0]
            # if args.dataset in ['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed']:
            #     graph = graph_ori.clone()
            #     deg = graph.remove_self_loop().in_degrees().float()
            # else:
            #     assert args.dataset in ['P50', 'P_20_50']
            #     features_ori, adjs_ori, triplets_ori = graph_ori
            #     num_node = features_ori.shape[0]
            #     deg = np.zeros(num_node)
            #     for i in range(10):      # Skip self-loop without counting the 10th
            #         new_deg = degree(adjs_ori[i].coalesce().indices()[0], num_nodes=num_node)
            #         deg = deg + new_deg
            # stability = (deg.std() / deg.mean()).item()
            tmp_stable_list.append(stability)
            tmp_perf_list.append(test_perf)
        
        stability_list.append(tmp_stable_list)
        perf_list.append(tmp_perf_list)
    
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(stability_list, columns=dataset_list, index=model_list),
        annot=perf_list, cmap="YlGnBu", fmt='.2f', )
    plt.savefig('explain.png', bbox_inches='tight')
    plt.savefig('explain.pdf', bbox_inches='tight')


def plot_curve(args):
    gt = load_results(args, 'brute')
    res = load_results(args, args.method)
    res[res < 0] = 0
    plt.figure(figsize=(10, 10))
    plt.plot(gt, res)
    plt.savefig('plot.png', bbox_inches='tight')


def arxiv_case_study(args):
    assert args.model == 'DrGAT'
    gt = load_results(args, 'brute')
    sort_idx = np.argsort(gt)
    idx2mag = pd.read_csv('arxiv/data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz')
    mag2title = pd.read_csv('arxiv/data/ogbn_arxiv/mapping/titleabs.tsv', sep='\t')
    label2category = pd.read_csv('arxiv/data/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz')
    data_ori = load_arxiv_dataset(data_dir='./arxiv/data')
    _, labels_ori = data_ori[0]
    graph_ori = load_dataset(args, device=device, run_time=0)

    ori_out_list = []
    for run_time in range(5):
        model = load_model(run_time, args, device, graph_ori)
        model.eval()
        with torch.no_grad():
            ori_out = model(graph_ori, graph_ori.ndata['feat'])
            ori_out = F.softmax(ori_out, dim=1)
            ori_out_list.append(ori_out)
    ori_out_list = torch.stack(ori_out_list)
    ori_out = ori_out_list.mean(0)

    for i in range(1, 11):
        r_node = sort_idx[-i]
        mag = idx2mag[idx2mag['node idx'] == r_node]['paper id']
        if int(mag) == 200971:
            title = 'ontology as a source for rule generation'
        else:
            title = mag2title[mag2title['200971'] == int(mag)]['ontology as a source for rule generation']
        category = label2category[label2category['label idx'] == labels_ori[r_node].item()]
        category = category['arxiv category'].values[0]
        print(f'Top {i}: {title.values[0]}\nCategory: {category}')

        graph = graph_ori.clone()
        graph.remove_nodes(r_node)
        new_out_list = []
        for run_time in range(5):
            model = load_model(run_time, args, device, graph_ori)
            model.eval()
            with torch.no_grad():
                out = model(graph, graph.ndata['feat'])
                out = F.softmax(out, dim=1)
                new_out_list.append(out)
        new_out = torch.stack(new_out_list).mean(0)
        bias = new_out - torch.cat((ori_out[:r_node], ori_out[r_node+1:]))
        bias = bias.mean(0)
        top_idx = torch.topk(bias.abs(), 3).indices
        for idx in top_idx:
            top_label = label2category[label2category['label idx'] == labels_ori[idx].item()]
            top_label = top_label['arxiv category'].values[0]
            print(f'{top_label} changed by {bias[idx]};', end=' ')
        print()


def twitter_case_study(args):
    assert args.model == 'TIMME'
    gt = load_results(args, 'brute')
    sort_idx = np.argsort(gt)
    twitter_ids = pd.read_csv(f'TIMME/data/{args.dataset}/all_twitter_ids.csv')['twitter_id']
    # user_info = json.load(open(f'TIMME/data/formatted_location/simplified_user_info.json'))
    # dict_info = pd.read_csv(f'TIMME/data/{args.dataset}/all_twitter_ids.csv')['twitter_id']
    adjs_ori, features_ori, labels_info, trainable, mask, link_info, (label_map, all_id_list) = \
        multi_relation_load(f'TIMME/data/{args.dataset}', 
        files=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], 
        feature_data='one_hot', feature_file=None, freeze_feature=False, split_links=False,
        additional_labels_files=["TIMME/data/additional_labels/new_dict_cleaned.csv"], cycle=0)
    idx_train, idx_val, idx_test, labels_ori = labels_info
    labels = labels.to(device)
    adjs_ori = [i.to(device) for i in adjs_ori]
    features_ori = features_ori.to(device)
    graph_ori = (features_ori, adjs_ori, None)
    num_node = features_ori.shape[0]

    ori_out_list = []
    for run_time in range(5):
        model = load_model(run_time, args, device, graph_ori)
        model.eval()
        with torch.no_grad():
            ori_out = model(features_ori, adjs_ori, only_classify=True)
            ori_out = torch.exp(ori_out)
            ori_out_list.append(ori_out)
    ori_out_list = torch.stack(ori_out_list)
    ori_out = ori_out_list.mean(0)

    for i in range(1, 11):
        r_node = sort_idx[-i]
        tid = twitter_ids[i]
        print(f'Top {i}: {tid}')
        new_out_list = []
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
        
        for run_time in range(5):
            model = load_model(run_time, args, device, graph_ori)
            model.eval()
            with torch.no_grad():
                out = model(features, adjs, only_classify=True)
                out = torch.exp(out)
                new_out_list.append(out)
        new_out = torch.stack(new_out_list).mean(0)
        bias = new_out - torch.cat((ori_out[:r_node], ori_out[r_node+1:]))
        bias = bias.mean(0)
        print(f'Label changed by {bias}')
        print()


# analyze how much the top influential nodes account for the total influence
def analyze_top_rate(args):
    rates = []
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        # model_list = ['GCN', 'GraphSAGE', 'GAT', 'GCNII']
        model_list = ['GCN_edge', 'GraphSAGE_edge', 'GAT_edge', 'GCNII_edge']
    elif args.dataset == 'ogbn-arxiv':
        # model_list = ['GCN', 'GraphSAGE', 'DrGAT', 'GCNII']
        model_list = ['GCN_edge', 'GraphSAGE_edge', 'GAT_edge', 'GCNII_edge']
    else:
        # model_list = ['TIMME']
        model_list = ['TIMME_edge']
    for model in model_list:
        tmp_rates = []
        args.model = model
        args.dropout = 0
        args.hidden_size = 0
        args.num_layers = 0
        args = load_default_args(args)
        gt = load_results(args, 'brute')
        gt = np.sort(gt)
        for i in [0.01, 0.03, 0.1]:
            tmp_rates.append(gt[round(-i * len(gt)):].sum() / gt.sum())
        rates.append(tmp_rates)
        print(np.mean(rates, 0))


# Analyze the relation between node influence and degree on Twitter datasets
def twitter_degree(args):
    adjs, features, labels_info, trainable, mask, link_info, (label_map, all_id_list) = \
        multi_relation_load(f'TIMME/data/{args.dataset}', 
        files=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], 
        feature_data='one_hot', feature_file=None, freeze_feature=False, split_links=False,
        additional_labels_files=["TIMME/data/additional_labels/new_dict_cleaned.csv"], cycle=0)
    num_nodes_dict = {'user': features.shape[0]}
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
    deg_in1 = graph.in_degrees(etype='r1').float()     # retweet
    deg_out1 = graph.out_degrees(etype='r1').float()
    deg_in3 = graph.in_degrees(etype='r3').float()     # mention
    deg_out3 = graph.out_degrees(etype='r3').float()
    deg_in5 = graph.in_degrees(etype='r5').float()     # friend
    deg_out5 = graph.out_degrees(etype='r5').float()
    deg_in7 = graph.in_degrees(etype='r7').float()     # reply
    deg_out7 = graph.out_degrees(etype='r7').float()
    deg_in9 = graph.in_degrees(etype='r9').float()     # favorite
    deg_out9 = graph.out_degrees(etype='r9').float()
    total_in = deg_in1 + deg_in3 + deg_in5 + deg_in7 + deg_in9
    total_out = deg_out1 + deg_out3 + deg_out5 + deg_out7 + deg_out9

    x = list(range(100))
    args.model = 'TIMME'
    gt1 = load_results(args, 'brute')
    args.model = 'TIMME_edge'
    gt2 = load_results(args, 'brute')
    gt = np.mean([gt1, gt2], 0)
    sort_idx = np.argsort(gt)
    in_list, out_list = [], []
    for i in x:
        hot_sort = sort_idx[round(i / len(x) * len(gt)): round((i + 1) / len(x) * len(gt))]
        in_list.append(total_in[hot_sort].mean())
        out_list.append(total_out[hot_sort].mean())

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_ylabel("Node degree", size=22)
    ax1.set_xlabel('Influence level', size=22)
    line1 = ax1.bar(x, out_list, label='out degree', color='blue')
    line2 = ax1.bar(x, in_list, bottom=out_list, label='in degree', color='skyblue')
    plt.yticks(size=16)

    ax2 = ax1.twinx()
    ax2.set_ylabel('In-degree / out-degree', size=22)
    line3, = ax2.plot(x, np.array(in_list) / np.array(out_list), label='in-degree / out-degree', color='red',
        linewidth=2)
    plt.legend(handles=[line1, line2, line3], prop={'size': 20})
    plt.yticks(size=18)
    plt.xticks([])
    plt.title(args.dataset, size=26)
    plt.savefig(f'degree_{args.dataset}.png', bbox_inches='tight')
    plt.savefig(f'degree_{args.dataset}.pdf', bbox_inches='tight')


def twitter_edge(args):
    adjs, features, labels_info, trainable, mask, link_info, (label_map, all_id_list) = \
        multi_relation_load(f'TIMME/data/{args.dataset}', 
        files=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], 
        feature_data='one_hot', feature_file=None, freeze_feature=False, split_links=False,
        additional_labels_files=["TIMME/data/additional_labels/new_dict_cleaned.csv"], cycle=0)
    num_nodes_dict = {'user': features.shape[0]}
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
    deg_in1 = graph.in_degrees(etype='r1').float()     # retweet
    deg_out1 = graph.out_degrees(etype='r1').float()
    deg_in3 = graph.in_degrees(etype='r3').float()     # mention
    deg_out3 = graph.out_degrees(etype='r3').float()
    deg_in5 = graph.in_degrees(etype='r5').float()     # friend
    deg_out5 = graph.out_degrees(etype='r5').float()
    deg_in7 = graph.in_degrees(etype='r7').float()     # reply
    deg_out7 = graph.out_degrees(etype='r7').float()
    deg_in9 = graph.in_degrees(etype='r9').float()     # favorite
    deg_out9 = graph.out_degrees(etype='r9').float()
    
    cnt_in1, cnt_out1, cnt_in3, cnt_out3, cnt_in5, cnt_out5, cnt_in7, cnt_out7, cnt_in9, \
        cnt_out9 = [], [], [], [], [], [], [], [], [], []
    args.model = 'TIMME'
    gt1 = load_results(args, 'brute')
    args.model = 'TIMME_edge'
    gt2 = load_results(args, 'brute')
    gt = np.mean([gt1, gt2], 0)
    sort_idx = np.argsort(gt)
    x = list(range(20))
    for i in x:
        hot_sort = sort_idx[round(i / len(x) * len(gt)): round((i + 1) / len(x) * len(gt))]
        cnt_in1.append(deg_in1[hot_sort].mean())
        cnt_out1.append(deg_out1[hot_sort].mean())
        cnt_in3.append(deg_in3[hot_sort].mean())
        cnt_out3.append(deg_out3[hot_sort].mean())
        cnt_in5.append(deg_in5[hot_sort].mean())
        cnt_out5.append(deg_out5[hot_sort].mean())
        cnt_in7.append(deg_in7[hot_sort].mean())
        cnt_out7.append(deg_out7[hot_sort].mean())
        cnt_in9.append(deg_in9[hot_sort].mean())
        cnt_out9.append(deg_out9[hot_sort].mean())

    cnt_in1 = [i / np.sum(cnt_in1) for i in cnt_in1]
    cnt_in3 = [i / np.sum(cnt_in3) for i in cnt_in3]
    cnt_in5 = [i / np.sum(cnt_in5) for i in cnt_in5]
    cnt_in7 = [i / np.sum(cnt_in7) for i in cnt_in7]
    cnt_in9 = [i / np.sum(cnt_in9) for i in cnt_in9]
    cnt_out1 = [i / np.sum(cnt_out1) for i in cnt_out1]
    cnt_out3 = [i / np.sum(cnt_out3) for i in cnt_out3]
    cnt_out5 = [i / np.sum(cnt_out5) for i in cnt_out5]
    cnt_out7 = [i / np.sum(cnt_out7) for i in cnt_out7]
    cnt_out9 = [i / np.sum(cnt_out9) for i in cnt_out9]

    plt.figure(figsize=(8, 8))
    plt.xlabel('Influence level', size=22)
    plt.ylabel('Edge ratio', size=22)
    plt.plot(x, cnt_in1, label='retweeted', color='green')
    plt.plot(x, cnt_in3, label='mentioned', color='orange')
    plt.plot(x, cnt_in5, label='followed', color='red')
    plt.plot(x, cnt_in7, label='replied', color='blue')
    plt.plot(x, cnt_in9, label='liked', color='indigo')
    plt.plot(x, cnt_out1, label='retweet', color='aquamarine')
    plt.plot(x, cnt_out3, label='mention', color='yellow')
    plt.plot(x, cnt_out5, label='follow', color='lightcoral')
    plt.plot(x, cnt_out7, label='reply', color='skyblue')
    plt.plot(x, cnt_out9, label='like', color='violet')
    plt.legend(prop={'size': 20})
    plt.title(args.dataset, size=26)
    plt.xticks([])
    plt.yticks(size=18)
    plt.savefig(f'relation_{args.dataset}.png', bbox_inches='tight')
    plt.savefig(f'relation_{args.dataset}.pdf', bbox_inches='tight')
    plt.close()


def main():
    argparser = argparse.ArgumentParser("Evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--dataset", type=str, default="Cora", choices=['Cora', 
        'CiteSeer', 'PubMed', 'ogbn-arxiv', 'P50', 'P_20_50'])
    argparser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT',
        'DrGAT', 'GCNII', 'TIMME', 'GCN_edge', 'GraphSAGE_edge', 'GAT_edge', 'GCNII_edge', 'TIMME_edge'])
    argparser.add_argument('--analyze', type=str, default='performance', choices=['performance',
        'stability', 'explain', 'plot_curve', 'case_study', 'top_rate', 'twitter_degree', 'twitter_edge'],
        help="performance: basic analysis; \
        stability: analyze stability of different models and hyper-parameters; \
        explain: explain the model performance \
        plot_curve: plot the predicted and real influence \
        case_study: observe the top influential nodes \
        top_rate: analyze how much the top influential nodes account for the total influence \
        twitter_degree: analyze the relationship between influence and degree on Twitter datasets \
        twitter_edge: analyze the relationship between influence and edge type on Twitter datasets")
    argparser.add_argument('--num_layers', type=int, default=0, help="0 means default")
    argparser.add_argument('--hidden_size', type=int, default=0, help="0 means default")
    argparser.add_argument('--dropout', type=int, default=0, help="0 means default")
    argparser.add_argument('--method', type=str, default='nora', choices=['nora', 'degree',
        'betweenness', 'mask', 'gcn-n', 'gcn-e', 'gat-n', 'gat-e'])
    argparser.add_argument('--train_ratio', type=float, default=0, help="for method 'gcn/gat'")
    argparser.add_argument('--val_ratio', type=float, default=0, help="for method 'mask' and 'gcn/gat'")
    args = argparser.parse_args()

    args = load_default_args(args)
    if args.analyze == 'performance':
        performance(args)
    elif args.analyze == 'stability':
        if args.dataset == 'ogbn-arxiv':
            stability_arxiv()
        elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            stability_planetoid()
    elif args.analyze == 'explain':
        explain(args)
    elif args.analyze == 'plot_curve':
        plot_curve(args)
    elif args.analyze == 'case_study':
        if args.dataset == 'ogbn-arxiv':
            arxiv_case_study(args)
        else:
            assert args.dataset in ['P50', 'P_20_50']
            twitter_case_study(args)
    elif args.analyze == 'top_rate':
        analyze_top_rate(args)
    elif args.analyze == 'twitter_degree':
        twitter_degree(args)
    elif args.analyze == 'twitter_edge':
        twitter_edge(args)


if __name__ == '__main__':
    main()
