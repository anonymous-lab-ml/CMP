import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "ERM",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["ERM"]},
        "lr": {"values":[2e-4]},
        
        "batch_size": {"values":[256]},
        "feature_dimension": {"values":[512]},
        "epochs": {"values":[20]},
        "seed": {"values":[1001,1002,1003,1004,1005]},
        "split_scheme": {"values": ["official"]},
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-RMNIST")
print(sweep_id)
wandb.agent(sweep_id)