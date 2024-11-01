import torch
import torch.autograd as autograd
import numpy as np
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.utils import split_into_groups
from wilds.common.grouper import CombinatorialGrouper

import wandb
from tqdm.auto import tqdm

from src.models import CNN, Classifier, ResNet18
from src.datasets import RotatedMNIST



class ERM(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.dataset = RotatedMNIST(root_dir=self.hparam["root_dir"], split_scheme=self.hparam["split_scheme"])
        # self.alter_dataset = get_dataset(dataset="celebA", root_dir=self.hparam["root_dir"], download=True)
        # print(self.alter_dataset.metadata_array.shape)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_set = self.dataset.get_subset(split='train')
        self.train_loader = get_train_loader(self.loader_type, self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=True, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.in_test_set = self.dataset.get_subset(split='in_test')
        self.in_test_loader = get_eval_loader(loader='standard', dataset=self.in_test_set, batch_size=self.hparam["batch_size"])
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        # self.featurizer = CNN(input_shape=(1,28,28), n_outputs=self.hparam['feature_dimension'])
        self.featurizer = ResNet18(input_shape=(1,28,28), n_outputs=self.hparam['feature_dimension'])
        self.classifier = Classifier(in_features=self.hparam['feature_dimension'], out_features=10, is_nonlinear=False)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            for x,y_true,metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y_true)
                with torch.no_grad():
                    total_loss += loss.item() * len(y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
    
    def evaluate(self, step):
        self.model.eval()
        in_corr = 0.
        for x,y_true,metadata in self.in_test_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            metadata = metadata.to(self.device)
            outputs = self.model(x)
            y_pred = torch.argmax(outputs, dim=-1)
            in_corr += torch.sum(torch.eq(y_pred, y_true))

        corr = 0.
        for x,y_true,metadata in self.test_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            metadata = metadata.to(self.device)
            outputs = self.model(x)
            y_pred = torch.argmax(outputs, dim=-1)
            corr += torch.sum(torch.eq(y_pred, y_true))
        if self.hparam['wandb']:
            wandb.log({"in_test_acc": in_corr / len(self.test_set), "test_acc": corr / len(self.test_set)}, step=step)
        else:
            print({"in_test_acc": in_corr / len(self.test_set), "test_acc": corr / len(self.test_set)})

    @property
    def loader_type(self):
        return 'standard'

    @property
    def domain_fields(self):
        return ['domain']

    @property
    def n_groups_per_batch(self):
        return 1


class CF_Pair(ERM):
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                
                _, group_indices, _ = split_into_groups(g)
                group_indices = self.form_group(group_indices)
                features = self.featurizer(x)
                outputs = self.classifier(features)
                features_0 = features[group_indices[0]]
                features_1 = features[group_indices[1]]
                penalty = self.distance(features_0, features_1) ** self.hparam["param2"]
                loss = self.criterion(outputs, y_true)
                # print(loss, penalty)
                obj = loss + self.hparam["param1"] * penalty
                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"erm_loss": total_celoss.item() / len(self.train_set), "cf_loss": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
        # self.scheduler.step()
    
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['id']
    
    @property
    def n_groups_per_batch(self):
        return int(self.hparam['batch_size'] / 2)
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')

    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T


    
class IRM(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.scale = torch.tensor(1.).to(self.device).requires_grad_()
        self.update_count = 0
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.penalty_anneal_iters = self.hparam["param2"]
        self.penalty_weight = self.hparam["param1"]
        self.update_count = 0.
    
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                
                _, group_indices, _ = split_into_groups(g)
                outputs = self.model(x)
                penalty = 0.
                loss = self.criterion(outputs*self.scale, y_true)
                penalty = self.irm_penalty(loss[group_indices[0]], loss[group_indices[1]])
                if self.update_count >= self.penalty_anneal_iters:
                    penalty_weight = self.penalty_weight
                else:
                    penalty_weight = self.update_count / self.penalty_anneal_iters
                avg_loss = loss.mean()
                obj = avg_loss + penalty_weight * penalty
                with torch.no_grad():
                    total_celoss += avg_loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (avg_loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_count += 1

            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)


    def irm_penalty(self, loss_0, loss_1):
        grad_0 = autograd.grad(loss_0.mean(), [self.scale], create_graph=True)[0]
        grad_1 = autograd.grad(loss_1.mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_0 * grad_1)
        del grad_0, grad_1
        return result
    
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['domain']
    
    @property
    def n_groups_per_batch(self):
        return 2

    def form_group(self, group_indices):
        return group_indices
    


class Fewshot(ERM):
    """
    param 1: number of fine tune sample.
    param 2: penalty for the alignment
    param 3: penalty for the fine-tune set weight.
    """
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.dataset = RotatedMNIST(root_dir=self.hparam["root_dir"], split_scheme=self.hparam["split_scheme"])
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_set = self.dataset.get_subset(split='train')
        self.fewshot_idx = np.random.choice(60000, int(self.hparam['param3']), replace=False)
        self.fewshot_iter = self.sample_u_from_n(self.fewshot_idx, self.hparam['fewshot_batch_size'])
        # for i in range(len(self.dataset._training_domains)):
        #     fewshot_idx = np.concatenate([fewshot_idx, idx + i * 60000])
        self.train_loader = get_train_loader('standard', self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=None)
        self.in_test_set = self.dataset.get_subset(split='in_test')
        self.in_test_loader = get_eval_loader(loader='standard', dataset=self.in_test_set, batch_size=self.hparam["batch_size"])
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        self.featurizer = CNN(input_shape=(1,28,28), n_outputs=self.hparam['feature_dimension'])
        self.classifier = Classifier(in_features=self.hparam['feature_dimension'], out_features=10)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            total_loss = 0.
            erm_loss = 0.
            cf_loss = 0.
            self.model.train()
            for x, y_true,metadata in tqdm(self.train_loader):
                batch_size = x.shape[0]
                idx = next(self.fewshot_iter)
                fewshot_idx = []
                for j in range(len(self.dataset._training_domains)):
                    fewshot_idx = np.concatenate([fewshot_idx, idx + j * 60000])
                fewshot_idx.astype(int)
                fewshot_idx = fewshot_idx.astype(np.int64, copy=False)
                fewshot_data = self.train_set[fewshot_idx][0]
                fewshot_target = self.train_set[fewshot_idx][1]
                fewshot_metadata = self.train_set[fewshot_idx][2]
                x = torch.cat((x, fewshot_data)).to(self.device)
                y_true = torch.cat((y_true, fewshot_target)).to(self.device)
                metadata = torch.cat((metadata, fewshot_metadata)).to(self.device)

                features = self.featurizer(x)
                outputs = self.classifier(features)

                feature_per_domain = features[batch_size:].reshape(len(self.dataset._training_domains), int(self.hparam['fewshot_batch_size']), int(self.hparam['feature_dimension']))
                feature_mean = feature_per_domain.mean(dim=0, keepdim=True)
                loss_2 = (torch.linalg.norm((feature_per_domain-feature_mean).flatten(), ord=2)) ** (2*self.hparam['param2']) / feature_per_domain.shape[1] / self.hparam['feature_dimension']
                loss = self.criterion(outputs[:batch_size], y_true[:batch_size])
                obj = loss + self.hparam['param1'] * loss_2
                with torch.no_grad():
                    total_loss += obj.item() * len(y_true)
                    erm_loss += loss.item() * len(y_true)
                    cf_loss += loss_2.item() * len(y_true)
                self.optimizer.zero_grad()
                obj.backward()
                self.optimizer.step()
                
            if self.hparam['wandb']:
                wandb.log({"train_loss": total_loss / len(self.train_set), "erm_loss": erm_loss / len(self.train_set), "cf_loss": cf_loss / len(self.train_set)}, step=i)
            else:
                print({"train_loss": total_loss / len(self.train_set)})
            self.evaluate(i)
        self.optimizer.zero_grad()
    
    @staticmethod
    def sample_u_from_n(arr, fewshot_batch_size):
        def iterator():
            # Shuffle the list first to ensure randomness
            np.random.shuffle(arr)
            i = 0
            while True:
                if i + fewshot_batch_size >= len(arr):
                    yield arr[i:]
                    np.random.shuffle(arr)
                    i = 0
                else:
                    yield arr[i:i+fewshot_batch_size]
                    i += fewshot_batch_size
        return iterator()

    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['id']
    
    @property
    def n_groups_per_batch(self):
        return int(self.hparam['batch_size'] / 2)
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')

    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T
    
