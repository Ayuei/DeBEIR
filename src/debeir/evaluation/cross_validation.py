"""
Cross validation object implementations for a dataset agnostic k-fold validation.

Input datasets types are defined in `debeir.core.datasets.types`
"""
from enum import Enum
from typing import Dict, List, Union

import datasets
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from debeir.datasets.types import DatasetTypes, InputExample


# noinspection PyTypeChecker
def split_k_fold(n_fold, data_files):
    """
    Split a csv dataset k times

    :param n_fold: Number of folds to produce
    :type n_fold:
    :param data_files: The file paths in csv format for splitting
    :type data_files:
    :return: Huggingface Dataset objects of the train and validation splits
    :rtype:
    """
    percentage = 100 // n_fold

    vals_ds = datasets.load_dataset('csv', split=[
        f'train[{k}%:{k + percentage}%]' for k in range(0, 100, percentage)
    ], data_files=data_files)

    trains_ds = datasets.load_dataset('csv', split=[
        f'train[:{k}%]+train[{k + percentage}%:]' for k in range(0, 100, percentage)
    ], data_files=data_files)

    return trains_ds, vals_ds


class CrossValidatorTypes(Enum):
    """
    Cross Validator Strategies for separating the dataset
    """
    Stratified = "StratifiedKFold"
    KFold = "KFold"


str_to_fn = {
    "StratifiedKFold": StratifiedKFold,
    "KFold": KFold
}


class CrossValidator:
    """
    Cross Validator Class for different types of data_sets

    E.g. List -> [[Data], label]
         List[Dict] -> {"data": Data, "label": label}
         Huggingface Dataset Object -> Data(set="train", label = "label").select(idx)
    """

    def __init__(self, dataset: Union[List, List[Dict], datasets.Dataset],
                 x_idx_label_or_attr: Union[str, int], y_idx_label_or_attr: Union[str, int],
                 cross_validator_type: [str, CrossValidatorTypes] = CrossValidatorTypes.Stratified,
                 seed=42, n_splits=5):
        # self.evaluator = evaluator
        self.cross_vali_fn = str_to_fn[cross_validator_type](n_splits=n_splits,
                                                             shuffle=True,
                                                             random_state=seed)
        self.dataset = dataset
        self.splits = []

        self.x_label = x_idx_label_or_attr
        self.y_label = y_idx_label_or_attr

        if self.dataset_type is None:
            self._determine_dataset_type()
            x, y = self.split_fn(x_idx_label_or_attr, y_idx_label_or_attr)
            self.splits = self.cross_vali_fn.split(x, y)

    def _determine_dataset_type(self):
        if isinstance(self.dataset, list):
            if isinstance(self.dataset[0], dict):
                self.dataset_type = DatasetTypes.ListDict
                self.split_fn = self._split_dict
            elif isinstance(self.dataset[0], InputExample):
                self.dataset_type = DatasetTypes.ListInputExample
                self.split_fn = self._split_list
            else:
                self.dataset_type = DatasetTypes.List
                self.split_fn = self._split_list
        elif isinstance(self.dataset, datasets.Dataset):
            self.dataset_type = DatasetTypes.HuggingfaceDataset
            self.split_fn = self._split_dataset
        else:
            raise NotImplementedError("Unknown Dataset format")

    def _split_list(self, *args, **kwargs):
        X = np.zeros(len(list(map(lambda k: k[self.x_label], self.dataset))))
        Y = map(lambda k: k[self.y_label], self.dataset)

        return X, Y

    def _split_dict(self, *args, **kwargs):
        X = np.zeros(len(list(map(lambda k: k[self.x_label], self.dataset))))
        Y = map(lambda k: k[self.y_label], self.dataset)

        return X, Y

    def _split_dataset(self, *args, **kwargs):
        # Rows data doesn't matter
        X = np.zeros(self.dataset.num_rows)
        Y = self.dataset[self.y_label]

        return X, Y

    def get_fold(self, fold_num: int):
        """

        :param fold_num: Which fold to pick
        :return:
        """

        split = self.splits[fold_num]

        return {
            "train_idxs": split[0],
            "val_idxs": split[1]
        }
