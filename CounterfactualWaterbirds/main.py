import argparse
import random
import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn
import wandb
from solver import *


def main(args):
    # hyperparameters
    hparam = vars(args)
    # setup WanDB
    if not args.no_wandb:
        wandb.init(project="CF_Waterbirds",
                    entity="anonymous-lab",
                    config=hparam)
        wandb.run.log_code()
    hparam['wandb'] = not args.no_wandb
    eval(hparam['solver'])(hparam).fit()

    

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain Generalization Pair')
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--root_dir', default='data', type=str)
    parser.add_argument('--dataset', default='waterbirds', type=str)
    parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--feature_dimension', default=1024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--param1', default=100, type=float)
    parser.add_argument('--param2', default=32, type=float)
    parser.add_argument('--param3', default=32, type=float)
    parser.add_argument('--solver', default='ERM', type=str)
    parser.add_argument('--upweighting', default="false", type=str)
    args = parser.parse_args()
    main(args)