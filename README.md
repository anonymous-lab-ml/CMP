# CMP
This is the anonymous code for submission of AISTATS 2025. Below is the script to run experiments.

## Synthetic
Run the jupyter notebook.

## RMNIST and Counterfactual-Waterbirds
It is simple to reproduce the experiment in the paper. Here's a demo:

```
cd ./RMNIST; python experiments/cf_fewshot.py
```
The Counterfactual-Waterbirds dataset can be found at: https://drive.google.com/file/d/1wmtPCdfT7oTbKyz7-sUUCPCEzSVfS2o6/view?usp=sharing

The environment:
```
conda env update -n my_env --file ENV.yaml
```
**Warning:** You may need to adjust the cuda version according to your own cuda version.
