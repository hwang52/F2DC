best_args = {
    'fl_digits': {
        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'f2dc': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'gum_tau': 0.1,
            'tem': 0.06,
            'agg_a': 1.0,
            'agg_b': 0.4,
            'lambda1': 0.8,
            'lambda2': 1.0
        }
    },
    'fl_officecaltech': {
        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'f2dc': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'gum_tau': 0.1,
            'tem': 0.06,
            'agg_a': 1.0,
            'agg_b': 0.4,
            'lambda1': 0.8,
            'lambda2': 1.0
        }
    },
    'fl_pacs': {
        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'f2dc': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'gum_tau': 0.1,
            'tem': 0.06,
            'agg_a': 1.0,
            'agg_b': 0.4,
            'lambda1': 0.8,
            'lambda2': 1.0
        }
    }
}