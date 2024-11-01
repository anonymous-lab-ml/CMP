from wilds.common.data_loaders import get_train_loader, get_eval_loader
from dataset import CounterfactualWaterbirdsDataset
import torchvision.transforms as transforms
import torch.nn as nn
from wilds.common.grouper import CombinatorialGrouper
from wilds import get_dataset
import torch
from models import ResNet50, Classifier
import wandb
from tqdm.auto import tqdm
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.metrics.loss import ElementwiseLoss
from wilds.common.utils import split_into_groups
import numpy as np
import torch.autograd as autograd


class ERM(object):
    def __init__(self, hparam):
        # hyperparameters
        self.hparam = hparam
        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # data
        if self.hparam['dataset'] == 'waterbirds':
            self.dataset = get_dataset('waterbirds', root_dir=hparam['root_dir'], download=False, split_scheme='official')
            self.train_set = self.dataset.get_subset('train', transform=self.train_transform)
        elif self.hparam['dataset'] == 'cfwaterbirds':
            self.dataset = CounterfactualWaterbirdsDataset(root_dir=hparam['root_dir'], download=False)
            split_mask = np.logical_or((self.dataset.split_array == 0), (self.dataset.split_array == 1))
            idx = np.where(split_mask)[0]
            self.train_set = WILDSSubset(self.dataset, idx, self.train_transform)
        self.test_set = self.dataset.get_subset('test', transform=self.eval_transform)
        self.train_loader = get_train_loader('standard', self.train_set, batch_size=self.hparam['batch_size'], num_workers=4, pin_memory=True)
        self.test_loader = get_eval_loader('standard', self.test_set, batch_size=64, num_workers=4, pin_memory=True)
        self.group_weight = torch.tensor([[1/3498, 1/56], [1/184, 1/1057]], dtype=torch.float32).to(self.device)
        # model
        self._featurizer = ResNet50(input_shape=(3,224,224), n_outputs=hparam['feature_dimension'])
        self._classifier = Classifier(in_features=hparam['feature_dimension'], out_features=2, is_nonlinear=False)
        self.featurizer = torch.nn.DataParallel(self._featurizer)
        self.classifier = torch.nn.DataParallel(self._classifier)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self._featurizer, self._classifier))
        self.model.to(self.device)
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparam['lr'], momentum=0.9, weight_decay=0.0001)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    @property
    def eval_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @property
    def train_transform(self):
        return transforms.Compose([
        transforms.RandomResizedCrop((224, 224),
        scale=(0.7, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
       
    def fit(self):
        for i in range(int(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            for x, y, meta in tqdm(self.train_loader): 
                x = x.to(self.device)
                y = y.to(self.device)
                meta = meta.to(self.device)
                train_y = self.model(x)
                if self.hparam['upweighting'] == 'true':
                    weight = self.group_weight[meta[:,0], meta[:,1]] 
                    loss = torch.sum(self.criterion(train_y, y) * weight) / weight.sum()
                elif self.hparam['upweighting'] == 'false':
                    loss = self.criterion(train_y, y).mean()                 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    # total_celoss += loss_train * len(y)
                    # total_penalty += loss_finetune * len(y)
                    total_loss += loss * len(y)
            if self.hparam['wandb']:
                # wandb.log({"erm_loss": total_celoss.item() / len(self.train_set), "cf_loss": total_penalty.item() / len(self.train_set), "training_loss": total_loss.item() / len(self.train_set)}, step=i)
                wandb.log({"training_loss": total_loss.item() / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)

    def evaluate(self, step):
        self.model.eval()
        y_pred = None
        for x, label, meta_batch in self.test_loader:
            x = x.to(self.device)
            label = label.to(self.device)
            meta_batch = meta_batch.to(self.device)
            outputs = self.model(x)
            prediction = torch.argmax(outputs, dim=-1)
            if y_pred is None:
                y_pred = prediction
                y_true = label
                metadata = meta_batch
            else:
                y_pred = torch.cat([y_pred, prediction])
                y_true = torch.cat([y_true, label])
                metadata = torch.cat([metadata, meta_batch])
        metric = self.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))

        if self.hparam['wandb']:
            wandb.log(metric[0], step=step)
        else:
            print(metric[0])


class CF_Pair(ERM):
    def __init__(self, hparam):
        super(CF_Pair, self).__init__(hparam)
        assert self.hparam['dataset'] == 'cfwaterbirds'
        self.fine_criterion = nn.MSELoss(reduction='mean')
        self.cf_set = self.dataset.get_subset('counterfactual', transform=self.train_transform)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['cf',])
        self.cf_loader = get_train_loader('group', self.cf_set, batch_size=int(hparam['param2']), num_workers=4, pin_memory=True, uniform_over_groups=None, grouper=self.grouper, distinct_groups=False, n_groups_per_batch=16)
        self.cf_iter = iter(self.cf_loader)
    
    
    def fit(self):
        for i in range(int(self.hparam['epochs'])):
            self.model.train()
            total_celoss = 0.
            total_penalty = 0.
            total_loss = 0.
            for x, y, meta in tqdm(self.train_loader):
                try:
                    finetune_x, finetune_y, finetune_meta = next(self.cf_iter)
                except StopIteration:
                    self.cf_iter = iter(self.cf_loader)
                    finetune_x, finetune_y, finetune_meta = next(self.cf_iter)
                
                x = x.to(self.device)
                y = y.to(self.device)
                meta = meta.to(self.device)

                finetune_x = finetune_x.to(self.device)
                finetune_y = finetune_y.to(self.device)
                finetune_meta = finetune_meta.to(self.device)

                finetune_len = finetune_meta.shape[0]

                all_x = torch.cat([x, finetune_x], dim=0)
                z_pred = self.featurizer(all_x)
                y_pred = self.classifier(z_pred)
                train_y = y_pred[:-finetune_len]
                fine_z = z_pred[-finetune_len:]
                if self.hparam['upweighting'] == 'true':
                    weight = self.group_weight[meta[:,0], meta[:,1]] 
                    loss_train = torch.sum(self.criterion(train_y, y) * weight) / weight.sum()
                elif self.hparam['upweighting'] == 'false':
                    loss_train = self.criterion(train_y, y).mean()            
                loss_finetune = self.fine_criterion(fine_z[0::2], fine_z[1::2])
                loss = loss_train + self.hparam['param1'] * loss_finetune
                with torch.no_grad():
                    total_celoss += loss_train * len(y)
                    total_penalty += loss_finetune * len(y)
                    total_loss += loss * len(y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.hparam['wandb']:
                wandb.log({"erm_loss": total_celoss.item() / len(self.train_set), "cf_loss": total_penalty.item() / len(self.train_set), "training_loss": total_loss.item() / len(self.train_set)}, step=i)
                # wandb.log({"training_loss": total_loss.item() / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))    
            self.evaluate(i)


class GroupDRO(ERM):
    """
    Group distributionally robust optimization.

    Original paper:
        @inproceedings{sagawa2019distributionally,
          title={Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle={International Conference on Learning Representations},
          year={2019}
        }    
    """
    def __init__(self, hparam):
        # initialize model
        super(GroupDRO, self).__init__(hparam)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['background','y'])
        self.train_loader = get_train_loader('group', self.train_set, batch_size=self.hparam['batch_size'], num_workers=4, pin_memory=True, uniform_over_groups=None, grouper=self.grouper, distinct_groups=False, n_groups_per_batch=2)
        # step size
        self.group_weights_step_size = self.hparam['param1'] # config.group_dro_step_size
        # initialize adversarial weights
        self.group_weights = torch.zeros(self.grouper.n_groups, device=self.device)
        train_g = self.grouper.metadata_to_group(self.train_set.metadata_array)
        unique_groups, unique_counts = torch.unique(train_g, sorted=False, return_counts=True)
        counts = torch.zeros(self.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
        self.loss = ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    

    def fit(self):
        for i in range(int(self.hparam['epochs'])):
            total_loss = 0.
            self.model.train()
            for x, y, meta in tqdm(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                g = self.grouper.metadata_to_group(meta).to(self.device)
                meta = meta.to(self.device)
                y_pred = self.model(x)
                group_losses, _, _ = self.loss.compute_group_wise(y_pred, y, g, self.grouper.n_groups, return_dict=False)
                loss = group_losses @ self.group_weights
                with torch.no_grad():
                    total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses.data)
                self.group_weights = (self.group_weights/(self.group_weights.sum()))
                self.optimizer.step()
                if self.hparam['wandb']:
                    wandb.log({"training_loss": total_loss / len(self.train_set)}, step=i)
                else:
                    print(total_loss / len(self.train_set))
            self.evaluate(i)


class IRM(ERM):
    def __init__(self, hparam):
        # initialize model
        super(IRM, self).__init__(hparam)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['background','y'])
        self.train_loader = get_train_loader('group', self.train_set, batch_size=self.hparam['batch_size'], num_workers=4, pin_memory=True, uniform_over_groups=None, grouper=self.grouper, distinct_groups=False, n_groups_per_batch=2)
        # step size
        self.loss = ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        self.penalty_weight = hparam['param1']
        self.penalty_anneal_iters = hparam['param2']
        self.scale = torch.tensor(1.).to(self.device).requires_grad_()
        self.update_count = 0


    def fit(self):
        for i in range(int(self.hparam['epochs'])):
            total_loss = 0.
            self.model.train()
            for x, y, meta in tqdm(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                g = self.grouper.metadata_to_group(meta).to(self.device)
                meta = meta.to(self.device)
                y_pred = self.model(x)
                unique_groups, group_indices, _ = split_into_groups(g)
                n_groups_per_batch = unique_groups.numel()
                avg_loss = 0.
                penalty = 0.
                for i_group in group_indices: # Each element of group_indices is a list of indices
                    group_losses, _ = self.loss.compute_flattened(y_pred[i_group] * self.scale, y[i_group], return_dict=False)
                    if group_losses.numel()>0:
                        avg_loss += group_losses.mean()
                    penalty += self.irm_penalty(group_losses)
                avg_loss /= n_groups_per_batch
                penalty /= n_groups_per_batch
                if self.update_count >= self.penalty_anneal_iters:
                    penalty_weight = self.penalty_weight
                else:
                    penalty_weight = self.update_count / self.penalty_anneal_iters
                penalty_weight = 0.
                # print(self.update_count, penalty_weight)
                objective = avg_loss + penalty * penalty_weight
                total_loss += objective.item()
                if self.update_count == self.penalty_anneal_iters:
                    # Reset Adam, because it doesn't like the sharp jump in gradient
                    # magnitudes that happens at this step.
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparam['lr'], momentum=0.9, weight_decay=0.0001)
                if objective.grad_fn is None:
                    pass
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                self.update_count += 1
            if self.hparam['wandb']:
                wandb.log({"loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))   
            self.evaluate(i)

    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        del grad_1, grad_2
        return result

