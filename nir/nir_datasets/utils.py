# TODO: Convert a Parser Return Dict (Dict[int, Dict[str, ...])
from collections import defaultdict
from typing import List, Union

import datasets
from datasets import DatasetDict

from evaluation.cross_validation import CrossValidator, DatasetTypes
from evaluation.evaluator import Evaluator
import string


class InputExample:
    """
    Copied from Sentence Transformer Library
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = '', texts: List[str] = None, label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label

        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))

    def get_label(self):
        return self.label

    def __getitem__(self, key):
        if key == "label":
            return self.get_label()

        if key == "texts":
            return self.texts

        if key in ["guid", "id"]:
            return self.guid

        raise KeyError()

    @classmethod
    def to_dict(cls, data: List['InputExample']):
        text_len = len(data[0].texts)
        processed_data = defaultdict(lambda: [])

        for datum in data:
            # string.ascii_lowercase

            processed_data["id"].append(datum.guid)
            processed_data["label"].append(datum.get_label())

            for i in range(text_len):
                letter = string.ascii_lowercase[i]  # abcdefghi
                # processed_data[text_a] = ...
                processed_data[f"text_{letter}"].append(datum.texts[i])

        return processed_data

    @classmethod
    def from_parser_output(cls, data):
        pass


class RelevanceExample(InputExample):
    """
    Converts Relevance Labels to 0 - 1
    """

    def __init__(self, max_score=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_score = max_score

    def get_label(self):
        return self.relevance()

    def relevance(self):
        """
        :return:
            Returns a normalised score for relevance between 0 - 1
        """
        return self.label / self.max_score


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
