import numpy as np
import torch
import torch.nn.functional as F
import dgl
# model
from planetoid.models import MyGCN, MyGraphSAGE, MyGAT
from planetoid.best_config import config as config_ori
from arxiv.DrGAT.model_drgat import DRGAT
from arxiv.simple.gnn import GCN_arxiv, GraphSAGE_arxiv
from arxiv.simple.gnn_edge import GCN_arxiv_edge
from TIMME.code.model.model import TIMME
# dataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ogb.nodeproppred import DglNodePropPredDataset
from TIMME.code.utils import multi_relation_load


# load dataset
def load_dataset(args, device, run_time=0):
    if args.dataset == 'ogbn-arxiv':
        data_ori = DglNodePropPredDataset(name="ogbn-arxiv", root='./arxiv/dataset')
        graph_ori, labels_ori = data_ori[0]
        if args.model == 'DrGAT':
            graph_ori = dgl.to_bidirected(graph_ori)
            graph_ori.ndata["feat"] = torch.from_numpy(np.load("./arxiv/dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy")).float()
        elif args.model in ['GCN', 'GraphSAGE', 'GCN-edge']:
            feat = graph_ori.ndata['feat']
            graph_ori = dgl.to_bidirected(graph_ori)
            graph_ori.ndata['feat'] = feat
        graph_ori = graph_ori.remove_self_loop().add_self_loop()
        graph_ori.create_formats_()
        graph_ori = graph_ori.to(device)
        if args.model == 'GCN-edge':
            test_neg = np.load(f'arxiv/dataset/ogbn_arxiv/processed/{run_time}_neg_link.npy')
            edge_index = [graph_ori.edges()[0].cpu().numpy(), graph_ori.edges()[1].cpu().numpy()]
            edge_index = np.array(edge_index)
            num_edge = edge_index.shape[1]
            test_mask = torch.zeros(num_edge, dtype=bool)
            num_test = num_edge - int(0.9 * num_edge)
            offset = int(run_time / 5 * num_edge)
            test_mask[min(int(0.9 * num_edge)+offset, num_edge): num_edge] = True
            test_mask[max(0, offset-num_test): max(0, offset)] = True
            test_link = torch.tensor(edge_index[:, test_mask])
            return (graph_ori, test_link.to(device), torch.tensor(test_neg).to(device)), None
        else:
            return graph_ori, None

    elif args.dataset in ['P50', 'P_20_50']:
        if args.model == 'TIMME':
            adjs_ori, features_ori, labels_info, trainable, mask, link_info, (label_map, all_id_list) = \
                multi_relation_load(f'TIMME/data/{args.dataset}', 
                files=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], 
                feature_data='one_hot', feature_file=None, freeze_feature=False, split_links=False,
                additional_labels_files=["TIMME/data/additional_labels/new_dict_cleaned.csv"], cycle=run_time)
        else:
            assert args.model == 'TIMME-edge'
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
        return (features_ori, adjs_ori, triplets_ori), None

    else:    # planetoid
        dataset = Planetoid(name=args.dataset, root='./planetoid/data')
        dataset.transform = T.NormalizeFeatures()
        data_ori = dataset.transform(dataset[0])     # make the features of a node to sum as 1
        data_ori.to(device)
        config = config_ori[args.model][args.dataset]
        if args.model == 'GCN-edge':
            neg_link = torch.tensor(np.load(f'planetoid/data/{args.dataset}/processed/{run_time}_neg_link.npy')).to(device)
            return [data_ori, neg_link], config
        else:
            return data_ori, config


def load_default_args(args, config):
    if args.n_layers == 0:
        if args.dataset == 'ogbn-arxiv':
            if args.model != 'GCN-edge':
                args.n_layers = 3
            else:
                args.n_layers = 2
        elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            args.n_layers = config['n_layers']
        else:
            args.n_layers = 2
    if args.hidden_size == 0:
        if args.dataset == 'ogbn-arxiv':
            if args.model != 'GCN-edge':
                args.hidden_size = 256
            else:
                args.hidden_size = 128
        elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            args.hidden_size = config['hidden_size']
    return args


def load_model(run_time, args, device, config, graph_ori):
    if args.dataset == 'ogbn-arxiv':
        if args.model == 'DrGAT':
            saved_name = f'arxiv/DrGAT/saved_model/{run_time}_student_{args.hidden_size}.pkl'
            model = DRGAT(768, 40, n_hidden=args.hidden_size, n_layers=3, n_heads=3, activation=F.relu,
                dropout=0.8, hid_drop=0.8, input_drop=0.35, attn_drop=0.0, edge_drop=0.5,
                use_attn_dst=False, use_symmetric_norm=True)
            model.load_state_dict(torch.load(saved_name))
        elif args.model in ['GCN', 'GraphSAGE']:
            saved_name = f'arxiv/simple/saved_model/{run_time}_{args.model.lower()}_{args.n_layers}_{args.hidden_size}.pkl'
            if args.model == 'GCN':
                model = GCN_arxiv(128, args.hidden_size, 40, args.n_layers, 0.5)
            elif args.model == 'GraphSAGE':
                model = GraphSAGE_arxiv(128, args.hidden_size, 40, args.n_layers, 0.5)
            model.load_state_dict(torch.load(saved_name))
        elif args.model == 'GCN-edge':
            saved_name = f'arxiv/simple/saved_model/{run_time}_gcn-edge_2_{args.hidden_size}.pkl'
            model = GCN_arxiv_edge(128, 128, 128, 2, 0.5)
            model.load_state_dict(torch.load(saved_name))
    
    elif args.dataset in ['P50', 'P_20_50']:
        features_ori, adjs_ori, triplets_ori = graph_ori
        model = TIMME(num_relation=5, num_entities=features_ori.shape[0], num_adjs=11,
            nfeat=features_ori.shape[1], nhid=100, nclass=2, dropout=0.1,
            relations=['retweet', 'mention', 'friend', 'reply', 'favorite'], regularization=0.01,
            skip_mode='add', attention_mode='self', trainable_features=None)
        saved_name = f'TIMME/saved_model/{run_time}_{args.dataset}_TIMME.pkl'
        model.load_state_dict(torch.load(saved_name))
    
    else:  # planetoid
        if args.model == 'GCN':
            model = MyGCN(in_channels=graph_ori.x.shape[1], out_channels=graph_ori.y.max().item()+1,
                hidden_channels=args.hidden_size, num_layers=args.n_layers, dropout=config['dropout'])
        elif args.model == 'GraphSAGE':
            model = MyGraphSAGE(in_channels=graph_ori.x.shape[1], out_channels=graph_ori.y.max().item()+1,
                hidden_channels=args.hidden_size, num_layers=args.n_layers, dropout=config['dropout'])
        elif args.model == 'GAT':
            model = MyGAT(in_channels=graph_ori.x.shape[1], out_channels=graph_ori.y.max().item()+1,
                hidden_channels=args.hidden_size, num_layers=args.n_layers, dropout=config['dropout'])
        elif args.model == 'GCN-edge':
            model = MyGCN(in_channels=graph_ori.x.shape[1], out_channels=128,
                hidden_channels=128, num_layers=2, dropout=config['dropout'])
        if args.hidden_size == config['hidden_size'] and args.model != 'GCN-edge':
            saved_name = f'planetoid/saved_model/{args.dataset}_{args.model}_{args.n_layers}_{run_time}.pkl'
        else:
            saved_name = f'planetoid/saved_model/{args.dataset}_{args.model}_2_128_{run_time}.pkl'
        model.load_state_dict(torch.load(f'{saved_name}'))

    print('loading model from', saved_name)
    model = model.to(device)
    return model
