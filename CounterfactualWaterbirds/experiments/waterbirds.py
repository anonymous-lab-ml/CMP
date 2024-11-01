import wandb
# todo here


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "Waterbirds-dataset",
    "metric": {
        "goal": "maximize",
        "name": "acc_wg"
        },
    "parameters": {
        "solver": {"values":["ERM"]},
        "dataset": {"values":["waterbirds", "cfwaterbirds"]},
        "lr": {"values":[1e-3]},
        "batch_size": {"values":[128]},
        "epochs": {"values":[300]},
        "seed": {"values":[1001, 1002, 1003, 1004, 1005]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CF_Waterbirds")
print(sweep_id)
wandb.agent(sweep_id)
