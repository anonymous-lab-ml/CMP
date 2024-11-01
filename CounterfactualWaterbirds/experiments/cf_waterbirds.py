import wandb
# todo here


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CF-Waterbirds-upweighting",
    "metric": {
        "goal": "maximize",
        "name": "acc_wg"
        },
    "parameters": {
        "solver": {"values":["ERM", "CF_Pair"]},
        "dataset": {"values":["cfwaterbirds"]},
        "upweighting": {"values":['false', 'true']},
        "param1": {"values":[500]},
        "param2": {"values":[256]},
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
