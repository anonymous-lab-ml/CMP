import wandb
# todo here


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CF-Waterbirds-irm",
    "metric": {
        "goal": "maximize",
        "name": "acc_wg"
        },
    "parameters": {
        "solver": {"values":["IRM"]},
        "dataset": {"values":["cfwaterbirds"]},
        "param1": {"values":[100]},
        "param2": {"values":[100]},
        "lr": {"values":[1e-3]},
        "feature_dimension": {"values":[64]},
        "batch_size": {"values":[128]},
        "epochs": {"values":[300]},
        "seed": {"values":[1001, 1002, 1003, 1004, 1005]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CF_Waterbirds")
print(sweep_id)
wandb.agent(sweep_id)
