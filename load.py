import numpy as np
import torch
import torch.nn.functional as F
import dgl
# model
from arxiv.DrGAT.model_drgat import DRGAT
from TIMME.code.model.model import TIMME
from planetoid.train import load_model as load_planetoid_model
from arxiv.others.train import load_model as load_arxiv_model
# dataset
from planetoid.train import load_dataset as load_planetoid_dataset
from arxiv.others.train import load_dataset as load_arxiv_dataset
from TIMME.code.utils import multi_relation_load
from planetoid.train import load_config as load_planetoid_config
from arxiv.others.train import load_config as load_arxiv_config


# load dataset
def load_dataset(args, device, run_time):
    if args.dataset == 'ogbn-arxiv':
        data_ori = load_arxiv_dataset(data_dir='./arxiv/data')
        graph_ori, labels_ori = data_ori[0]
        edge_index = torch.stack([graph_ori.edges()[0], graph_ori.edges()[1]])
        if args.model == 'DrGAT':
            graph_ori = dgl.to_bidirected(graph_ori)
            graph_ori.ndata["feat"] = torch.from_numpy(np.load("./arxiv/data/ogbn-arxiv-pretrain/X.all.xrt-emb.npy")).float()
        else:
            feat = graph_ori.ndata['feat']
            graph_ori = dgl.to_bidirected(graph_ori)
            graph_ori.ndata['feat'] = feat
        graph_ori = graph_ori.to(device)
        graph_ori = graph_ori.remove_self_loop().add_self_loop()
        graph_ori.create_formats_()
        if args.model[-4:] != 'edge':
            return graph_ori
        else:
            train_neg = np.load(f'arxiv/data/ogbn_arxiv/processed/{run_time}_train_neg_link.npy')
            val_neg = np.load(f'arxiv/data/ogbn_arxiv/processed/{run_time}_val_neg_link.npy')
            test_neg = np.load(f'arxiv/data/ogbn_arxiv/processed/{run_time}_neg_link.npy')
            neg_link = torch.concat([torch.tensor(train_neg), torch.tensor(val_neg), torch.tensor(test_neg)], 1)
            mask = (edge_index[1] >= edge_index[0])
            edge_index = edge_index[:, mask]
            assert edge_index.shape[1] == neg_link.shape[1]
            edge_index = torch.concat([edge_index, neg_link], 1).to(device)
            return (graph_ori, edge_index)

    elif args.dataset in ['P50', 'P_20_50']:
        if args.model[-4:] != 'edge':
            adjs_ori, features_ori, labels_info, trainable, mask, link_info, (label_map, all_id_list) = \
                multi_relation_load(f'TIMME/data/{args.dataset}', 
                files=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], 
                feature_data='one_hot', feature_file=None, freeze_feature=False, split_links=False,
                additional_labels_files=["TIMME/data/additional_labels/new_dict_cleaned.csv"], cycle=run_time)
        else:
            adjs_ori, features_ori, labels_info, trainable, mask, link_info, (label_map, all_id_list) = \
                multi_relation_load(f'TIMME/data/{args.dataset}', 
                files=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], 
                feature_data='one_hot', feature_file=None, freeze_feature=False, split_links=True,
                additional_labels_files=["TIMME/data/additional_labels/new_dict_cleaned.csv"], cycle=run_time)
        adjs_ori = [i.to(device) for i in adjs_ori]
        features_ori = features_ori.to(device)
        triplets_ori = []
        for i in range(5):
            triplets_ori.append(np.load(f'TIMME/data/{args.dataset}/{run_time}_sampled_triplets_{i}.npy'))
        return (features_ori, adjs_ori, triplets_ori)

    else:    # planetoid
        data_ori = load_planetoid_dataset(args, data_dir='./planetoid/data')
        if args.model[-4:] != 'edge':
            return data_ori
        else:
            train_neg = np.load(f'planetoid/data/{args.dataset}/{run_time}_train_neg_link.npy')
            val_neg = np.load(f'planetoid/data/{args.dataset}/{run_time}_val_neg_link.npy')
            test_neg = np.load(f'planetoid/data/{args.dataset}/{run_time}_neg_link.npy')
            graph_ori = data_ori[0]
            neg_link = torch.concat([torch.tensor(train_neg), torch.tensor(val_neg), torch.tensor(test_neg)], 1)
            edge_index = torch.stack([graph_ori.edges()[0], graph_ori.edges()[1]])
            mask = (edge_index[1] >= edge_index[0])
            edge_index = edge_index[:, mask]
            assert edge_index.shape[1] == neg_link.shape[1]
            edge_index = torch.concat([edge_index, neg_link], 1)
            return (data_ori, edge_index.to(device))


def load_default_args(args):
    if args.dataset == 'ogbn-arxiv':
        args = load_arxiv_config(args)
    elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        args = load_planetoid_config(args)
    else:   # Twitter datasdets
        if args.num_layers == 0:
            args.num_layers = 2
    if hasattr(args, 'cycles'):
        if args.cycles == []:
            args.cycles = [0, 1, 2, 3, 4]
    return args


def load_model(run_time, args, device, graph_ori):
    if args.dataset == 'ogbn-arxiv':
        if args.model == 'DrGAT':
            model = DRGAT(768, 40, n_hidden=args.hidden_size, n_layers=3, n_heads=3, activation=F.relu,
                dropout=0.8, hid_drop=0.8, input_drop=0.35, attn_drop=0.0, edge_drop=0.5,
                use_attn_dst=False, use_symmetric_norm=True)
            saved_name = f'arxiv/DrGAT/saved_model/{run_time}_student_{args.hidden_size}.pkl'
        else:
            model = load_arxiv_model(args, graph_ori)
            saved_name = f'arxiv/others/saved_model/{run_time}_{args.model.lower()}_{args.num_layers}_{args.hidden_size}.pkl'
    elif args.dataset in ['P50', 'P_20_50']:
        features_ori, adjs_ori, triplets_ori = graph_ori
        model = TIMME(num_relation=5, num_entities=features_ori.shape[0], num_adjs=11,
            nfeat=features_ori.shape[1], nhid=100, nclass=2, dropout=0.1,
            relations=['retweet', 'mention', 'friend', 'reply', 'favorite'], regularization=0.01,
            skip_mode='add', attention_mode='self', trainable_features=None)
        saved_name = f'TIMME/saved_model/{run_time}_{args.dataset}_TIMME.pkl'
    else:  # planetoid
        model = load_planetoid_model(args, graph_ori)
        saved_name = f'planetoid/saved_model/{run_time}_{args.dataset}_{args.model.lower()}_{args.num_layers}_{args.hidden_size}.pkl'
    model.load_state_dict(torch.load(saved_name))
    model = model.to(device)
    print('Loaded model from', saved_name)
    return model


# Load the saved results
def load_results(args, method):
    save_name = f'./save/{args.dataset}_{args.model}_{method}'
    if args.model not in ['TIMME', 'TIMME_edge']:
        save_name += f'_{args.num_layers}_{args.hidden_size}'
    res = np.load(f'{save_name}.npy', allow_pickle=True)
    assert len(res) == 5
    return res.mean(0)
