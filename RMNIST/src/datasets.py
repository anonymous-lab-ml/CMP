from wilds.datasets.wilds_dataset import WILDSDataset
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import os
import numpy as np
import copy
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from itertools import permutations, combinations_with_replacement
import random


class RotatedMNIST(WILDSDataset):
    _dataset_name = "rotatedMNIST"
    
    def __init__(
        self, version: str = None, root_dir: str = "data", 
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (28, 28)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(root_dir)
        # The original dataset contains 7 categories. 
        # if self._split_scheme == 'official':
        #     metadata_filename = "metadata.csv"
        #     print('dcc')
        # else:
        #     raise Not
        self._n_classes = 10
        self._angle_to_domain = {"0": 0, "15": 1, "30": 2, "45": 3, "60": 4, "75": 5, "90": 6}
        self._split_dict = {'train': 0, 'in_test': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'in_test': 'Test (In-Domain)', 'test': 'Test (OOD/Trans)'}
        self._metadata_fields = ["domain","angle", "y", "id", "idx"]
        self._split_array, self._data, self._y_array, self._metadata_array = self._get_data()
        
    def _get_data(self):
        # split scheme is the test domain rotation angle. We always test on test dataset.
        if self.split_scheme == "official":
            self._training_domains = ["0", "45"]
            self._in_test_domain = self._training_domains
            self._test_domain = "90"
        elif self.split_scheme == "oracle":
            self._training_domains = ["90"]
            self._in_test_domain = self._training_domains
            self._test_domain = "90"
        else: 
            raise NotImplementedError
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=True, download=True, transform=transforms.ToTensor()), batch_size=60000, shuffle=False)

        for x, y in train_loader:
            mnist_imgs = x
            mnist_labels = y
        mnist_x_list = []
        mnist_y_list = []
        metaarray_list = []
        mnist_split_list = []
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        idx = 0
        for angle in self._training_domains:
            for i in range(len(mnist_imgs)):
                if angle == '0':
                    mnist_x_list.append(to_tensor(to_pil(mnist_imgs[i])))
                else:
                    mnist_x_list.append(to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), int(angle))))
                d = self._angle_to_domain[angle]
                mnist_y_list.append(mnist_labels[i].item())
                metaarray_list.append(torch.tensor([d, int(angle), mnist_labels[i].item(), i, idx]))
                mnist_split_list.append(0)
                idx += 1

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=False, download=True, transform=transforms.ToTensor()), batch_size=10000, shuffle=False)
        for x, y in test_loader:
            mnist_imgs = x
            mnist_labels = y
        for i in range(len(mnist_imgs)):
            mnist_x_list.append(to_tensor(to_pil(mnist_imgs[i])))
            d = 0
            mnist_y_list.append(mnist_labels[i].item())
            metaarray_list.append(torch.tensor([d, 0, mnist_labels[i].item(), i, idx]))
            mnist_split_list.append(1)
            idx += 1
        
        for i in range(len(mnist_imgs)):
            if self._test_domain == '0':
                mnist_x_list.append(to_tensor(to_pil(mnist_imgs[i])))
            else:
                mnist_x_list.append(to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), int(self._test_domain))))
            d = self._angle_to_domain[self._test_domain]
            mnist_y_list.append(mnist_labels[i].item())
            metaarray_list.append(torch.tensor([d, int(angle), mnist_labels[i].item(), i, idx]))
            mnist_split_list.append(2)
            idx += 1

        # Stack
        img_array = torch.cat(mnist_x_list)
        y_array = torch.tensor(mnist_y_list)
        meta_array = torch.stack(metaarray_list, dim=0)
        split_array = torch.tensor(mnist_split_list)
    

        return split_array, img_array.unsqueeze(1), y_array, meta_array

    def get_input(self, idx) -> str:
        return self._data[idx]
