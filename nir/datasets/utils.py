# TODO: Convert a Parser Return Dict (Dict[int, Dict[str, ...])

import datasets

from nir.datasets.types import InputExample
from nir.evaluation.cross_validation import CrossValidator
from datasets.types import DatasetTypes
from nir.evaluation.evaluator import Evaluator


class CrossValidatorDataset:
    """
    Cross Validator Dataset
    """
    cross_val_cls: CrossValidator

    def __init__(self, dataset, cross_validator, n_folds, x_attr='text', y_attr='label'):
        self.cross_val_cls = cross_validator
        self.dataset = dataset
        self.fold = 0
        self.n_folds = n_folds
        self.x_attr = x_attr
        self.y_attr = y_attr
        self.folds = []

    @classmethod
    def prepare_cross_validator(cls, data, evaluator: Evaluator,
                                n_splits: int, x_attr, y_attr, seed=42) -> 'CrossValidatorDataset':
        """
        Prepare the cross validator dataset object that will internally produce the folds.

        :param data: Dataset to be used. Should be a list of dicts, or list of [x,y] or a Dataset object from datasets
        :param evaluator: Evaluator to use for checking results
        :param n_splits: Number of cross validation splits, k-fold (stratified)
        :param seed: Seed to use (default 42)
        :param y_attr: Label, or idx of the y label
        :param x_attr: Label or idx of the x label (not directly used)
        """

        return cls(data, CrossValidator(evaluator, data, x_attr, y_attr,
                                        n_splits=n_splits, seed=seed),
                   x_attr=x_attr, y_attr=y_attr,
                   n_folds=n_splits)

    def get_fold(self, idx) -> datasets.DatasetDict:
        """

        Get the fold and returns a dataset.DataDict object with
        DataDict{'train': ..., 'val': ...}

        :param idx:
        """

        train_idxs, val_idxs = self.cross_val_cls.get_fold(idx)
        dataset_dict = DatasetDict()

        if self.cross_val_cls.dataset_type in [DatasetTypes.List, DatasetTypes.ListDict]:
            # TODO: figure out how to make this into a huggingface dataset object generically
            train_subset = [self.dataset[i] for i in train_idxs]
            val_subset = [self.dataset[i] for i in val_idxs]
        elif self.cross_val_cls.dataset_type == DatasetTypes.ListInputExample:
            train_subset = InputExample.to_dict([self.dataset[i] for i in train_idxs])
            val_subset = InputExample.to_dict([self.dataset[i] for i in val_idxs])

            dataset_dict['train'] = datasets.Dataset.from_dict(train_subset)
            dataset_dict['val'] = datasets.Dataset.from_dict(val_subset)

        elif self.cross_val_cls.dataset_type == DatasetTypes.HuggingfaceDataset:
            train_subset = self.dataset.select(train_idxs)
            val_subset = self.dataset.select(val_idxs)

            dataset_dict['train'] = datasets.Dataset.from_dict(train_subset)
            dataset_dict['val'] = datasets.Dataset.from_dict(val_subset)

        return dataset_dict
