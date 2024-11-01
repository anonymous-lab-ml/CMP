import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy


class CounterfactualWaterbirdsDataset(WILDSDataset):
    _dataset_name = 'counterfactual-waterbirds'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/', # to update
            'compressed_size': None}}
    

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        # Note: metadata_df is one-indexed.
        metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))
        
        # Get the y values
        self._y_array = torch.LongTensor(metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = 2

        cf_values = np.nan_to_num(metadata_df['cf'].values, nan=0)
        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df['place'].values), self._y_array, torch.LongTensor(cf_values)),
            dim=1
        )
        self._metadata_fields = ['background', 'y', 'cf']
        self._metadata_map = {
            'background': ['land', 'water', 'snow', 'desert'], # Padding for str formatting
            'y': [' landbird', 'waterbird']
        }

        # Splits
        self._split_list = ['train', 'counterfactual', 'test', 'val']
        self._split_dict = {'train': 0, 'counterfactual': 1, 'test': 2, 'val': 3}
        self._split_names = {'train': 'Train', 'counterfactual': 'Counterfactual (ID)',
                             'test': 'Test (OOD)', 'val': 'Validation'}

        metadata_df['split'] = metadata_df['split_id'].apply(lambda x: self._split_list[x])
        self._split_array = metadata_df['split_id'].values


        # Extract filenames
        self._input_array = metadata_df['img_filename'].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['background', 'y']))
        
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       img_filename = os.path.join(
           self.data_dir,
           self._input_array[idx])
       x = Image.open(img_filename).convert('RGB')
       return x

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)

        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        # For Waterbirds, the validation and test sets are constructed to be more balanced
        # compared to the training set.
        # To compute the actual average accuracy over the empirical (training) distribution,
        # we therefore weight each groups according to their frequency in the training set.

        results['adj_acc_avg'] = (
            (results['acc_y:landbird_background:land'] * 3498
            + results['acc_y:landbird_background:water'] * 184
            + results['acc_y:waterbird_background:land'] * 56
            + results['acc_y:waterbird_background:water'] * 1057) /
            (3498 + 184 + 56 + 1057))

        results_str = f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n" + '\n'.join(results_str.split('\n')[1:])

        return results, results_str