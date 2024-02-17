planetoid_config = {
    'GCN':{
        'Cora':{
            'lr': 1e-2,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 1024,
            'dropout': 0.5
        },
        'CiteSeer':{
            'lr': 1e-2,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 1024,
            'dropout': 0.5
        },
        'PubMed':{
            'lr': 1e-2,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.5
        }
    },
    'GraphSAGE':{
        'Cora':{
            'lr': 1e-2,
            'wd': 0,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.5
        },
        'CiteSeer':{
            'lr': 5e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.5
        },
        'PubMed':{
            'lr': 1e-2,
            'wd': 1e-4,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.5
        }
    },
    'GAT':{
        'Cora':{
            'lr': 1e-2,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.5
        },
        'CiteSeer':{
            'lr': 1e-2,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.5
        },
        'PubMed':{
            'lr': 1e-2,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.5
        }
    },
    'GCN_edge':{
        'Cora':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 2048,
            'dropout': 0.7
        },
        'CiteSeer':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 2048,
            'dropout': 0.7
        },
        'PubMed':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 1024,
            'dropout': 0.7
        }
    },
    'GraphSAGE_edge':{
        'Cora':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.9
        },
        'CiteSeer':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 512,
            'dropout': 0.9
        },
        'PubMed':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 1024,
            'dropout': 0.9
        }
    },
    'GAT_edge':{
        'Cora':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 2048,
            'dropout': 0.9
        },
        'CiteSeer':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 2048,
            'dropout': 0.9
        },
        'PubMed':{
            'lr': 1e-3,
            'wd': 1e-5,
            'num_layers': 2,
            'hidden_size': 1024,
            'dropout': 0.5
        }
    },
}
