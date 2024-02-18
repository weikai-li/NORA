import numpy as np
import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
from load import load_default_args, load_results


def performance(args):
    gt = load_results(args, 'brute')
    res = load_results(args, args.method)
    # print(gt)
    # print(res)
    # exit()
    gt_sort = np.argsort(gt)
    res_sort = np.argsort(res)
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


def main():
    argparser = argparse.ArgumentParser("Evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--dataset", type=str, default="Cora", choices=['Cora', 
        'CiteSeer', 'PubMed', 'ogbn-arxiv', 'P50', 'P_20_50'])
    argparser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT',
        'DrGAT', 'TIMME', 'GCN_edge', 'GraphSAGE_edge', 'GAT_edge', 'TIMME_edge'])
    argparser.add_argument('--analyze', type=str, default='performance',
        choices=['performance', 'stability'],
        help="performance: basic analysis; \
        stability: analyze stability of different models and hyper-parameters")
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


if __name__ == '__main__':
    main()
