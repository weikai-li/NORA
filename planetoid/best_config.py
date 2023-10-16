config = {
    'GCN-edge':{
        'Cora':{
            'lr': 1e-2,
            'wd': 1e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 128,
            'dropout': 0.6,
            'n_epochs': 200,
            'patience': 20
        },
        'CiteSeer':{
            'lr': 1e-2,
            'wd': 3e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 128,
            'dropout': 0.5,
            'n_epochs': 200,
            'patience': 50
        },
        'PubMed':{
            'lr': 1e-2,
            'wd': 2e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 128,
            'dropout': 0.5,
            'n_epochs': 200,
            'patience': 20
        }
    },
    'GCN':{
        'Cora':{
            'lr': 1e-2,
            'wd': 1e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 1024,
            'dropout': 0.6,
            'n_epochs': 50,
            'patience': 20
        },
        'CiteSeer':{
            'lr': 1e-2,
            'wd': 3e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 1024,
            'dropout': 0.5,
            'n_epochs': 200,
            'patience': 50
        },
        'PubMed':{
            'lr': 1e-2,
            'wd': 2e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 1024,
            'dropout': 0.5,
            'n_epochs': 50,
            'patience': 20
        }
    },
    'GraphSAGE':{
        'Cora':{
            'lr': 1e-2,
            'wd': 1e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 256,
            'dropout': 0.9,
            'n_epochs': 100,
            'patience': 50
        },
        'CiteSeer':{
            'lr': 1e-2,
            'wd': 1e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 128,
            'dropout': 0.9,
            'n_epochs': 100,
            'patience': 50
        },
        'PubMed':{
            'lr': 1e-2,
            'wd': 1e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 256,
            'dropout': 0.5,
            'n_epochs': 150,
            'patience': 70
        }
    },
    'GAT':{
        'Cora':{
            'lr': 1e-2,
            'wd': 3e-5,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 1024,
            'dropout': 0.5,
            'n_epochs': 200,
            'patience': 100
        },
        'CiteSeer':{
            'lr': 1e-2,
            'wd': 1e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 256,
            'dropout': 0.5,
            'n_epochs': 150,
            'patience': 70
        },
        'PubMed':{
            'lr': 3e-3,
            'wd': 4e-4,
            'n_layers': 2,
            'data_transform': True,
            'hidden_size': 1024,
            'dropout': 0.3,
            'n_epochs': 150,
            'patience': 70
        }
    }
}

