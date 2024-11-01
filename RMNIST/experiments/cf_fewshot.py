import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CF-Fewshot",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["Fewshot"]},
        "param1": {"values":[0.1]},
        "param2": {"values":[1]},
        "param3": {"values":[1024, 2048, 4096, 8192, 16384, 60000]},
        "lr": {"values":[1e-3]},
        "feature_dimension": {"values":[10]},
        "batch_size": {"values":[512]},
        "fewshot_batch_size": {"values":[256]},
        "epochs": {"values":[20]},
        "seed": {"values":[1001, 1002, 1003, 1004, 1005]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-RMNIST")
print(sweep_id)
wandb.agent(sweep_id)
    