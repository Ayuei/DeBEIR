"""
Trainer Classes that are intended for use in hyperparameter tuning. Otherwise, for other encoding use cases, use the
`debeir.rankers.transformer_sent_encoder`
"""

import abc
from collections import defaultdict
from functools import partial
from typing import Dict, Sequence, Union

import loguru
import optuna
import torch
import torch_optimizer
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from wandb import wandb

from debeir.training.hparm_tuning.config import HparamConfig
from debeir.training.hparm_tuning.types import Hparam
from debeir.training.utils import LoggingEvaluator, LoggingLoss


class OptimizersWrapper:
    """This abstraction, allows for consistent API access to optimizer libraries: torch, torch_optimizer"""

    def __getattr__(self, name):
        if name in torch.optim.__dict__:
            return getattr(torch.optim, name)
        elif name in torch_optimizer.__dict__:
            return getattr(torch_optimizer, name)
        else:
            raise ModuleNotFoundError("Optimizer is not implemented, doesn't exist or is not supported.")


class Trainer:
    """
    Wrapper class for a trainer class.

    """

    def __init__(self, model, evaluator_fn, dataset_loading_fn):
        self.evaluator_fn = evaluator_fn
        self.model_cls = model  # Trainer object or method we will initialize
        self.dataset_loading_fn = dataset_loading_fn

    @abc.abstractmethod
    def fit(self, in_trial: optuna.Trial, train_dataset, val_dataset):
        raise NotImplementedError()


class SentenceTransformerHparamTrainer(Trainer):
    " See Optuna documentation for types! "
    model: SentenceTransformer

    def __init__(self, dataset_loading_fn, evaluator_fn, hparams_config: HparamConfig):
        super().__init__(SentenceTransformer, evaluator_fn, dataset_loading_fn)
        self.loss_fn = None
        self.hparams = hparams_config.parse_config_to_py() if hparams_config else None

    # noinspection PyUnresolvedReferences
    def get_optuna_hparams(self, trial: optuna.Trial, hparams: Sequence[Hparam] = None):
        """
        Get hyperparameters suggested by the optuna library

        :param trial: The optuna trial object
        :param hparams: Optional, pass a dictionary of HparamType[Enum] objects
        :return:
        """

        loguru.logger.info("Fitting the trainer.")

        hparam_values = defaultdict(lambda: 0.0)

        hparams = hparams if hparams else self.hparams

        if hparams is None:
            raise RuntimeError("No hyperparameters were specified")

        for key, hparam in hparams.items():
            if hasattr(hparam, 'suggest'):
                hparam_values[hparam.name] = hparam.suggest(trial)
                loguru.logger.info(f"Using {hparam_values[hparam.name]} for {hparam.name}.")
            else:
                hparam_values[key] = hparam

        return hparam_values

    def build_kwargs_and_model(self, hparams: Dict):
        kwargs = {}

        for hparam, hparam_value in list(hparams.items()):
            loguru.logger.info(f"Building model with {hparam}: {hparam_value}")

            if hparam == "lr":
                kwargs["optimizer_params"] = {
                    "lr": hparam_value
                }
            elif hparam == "model_name":
                self.model = self.model_cls(hparam_value)
            elif hparam == "optimizer":
                kwargs["optimizer_class"] = getattr(OptimizersWrapper(), hparam_value)
            elif hparam == "loss_fn":
                self.loss_fn = getattr(losses, hparam_value)
            else:
                kwargs[hparam] = hparam_value

        return kwargs

    def prehook(self, **kwargs):
        pass

    def fit(self, in_trial: optuna.Trial, train_dataset, val_dataset):
        """
        Fit the trainer model with the specified hyperpameters produced by the optuna trial.

        :param in_trial:
        :type in_trial:
        :param train_dataset:
        :type train_dataset:
        :param val_dataset:
        :type val_dataset:
        :return:
        :rtype:
        """
        hparams = self.get_optuna_hparams(in_trial)
        kwargs = self.build_kwargs_and_model(hparams)

        evaluator = self.evaluator_fn.from_input_examples(val_dataset)
        loss = self.loss_fn(model=self.model)
        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      batch_size=int(kwargs.pop("batch_size")), drop_last=True)

        self.prehook(**kwargs)

        # noinspection PyTypeChecker
        self.model.fit(
            train_objectives=[(train_dataloader, loss)],
            **kwargs,
            evaluator=evaluator,
            use_amp=True,
            callback=partial(trial_callback, in_trial)
        )

        return self.model.evaluate(evaluator)


def trial_callback(trial, score, epoch, *args, **kwargs):
    """
    Trial callback used by optuna. Records the scores for each epoch.

    :param trial:
    :type trial:
    :param score:
    :type score:
    :param epoch:
    :type epoch:
    :param args:
    :type args:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    trial.report(score, epoch)
    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


class SentenceTransformerTrainer(SentenceTransformerHparamTrainer):
    def __init__(self, dataset: Union[DatasetDict, Dict[str, Dataset]], hparams_config: HparamConfig,
                 evaluator_fn=None, evaluator=None, use_wandb=False):
        super().__init__(None, evaluator_fn, hparams_config)
        self.evaluator = evaluator
        self.use_wandb = use_wandb
        self.dataset = dataset

    def prehook(self, **kwargs):
        pass

    def fit(self, **extra_kwargs):
        kwargs = self.build_kwargs_and_model(self.hparams)

        if not self.evaluator:
            self.evaluator = LoggingEvaluator(self.evaluator_fn.from_input_examples(self.dataset['val']))

        loss = self.loss_fn(model=self.model)

        if self.use_wandb:
            wandb.watch(self.model)
            loss = LoggingLoss(loss)

        train_dataloader = DataLoader(self.dataset['train'], shuffle=True,
                                      batch_size=int(kwargs.pop("batch_size")),
                                      drop_last=True)

        self.prehook(**kwargs)

        self.model.fit(
            train_objectives=[(train_dataloader, loss)],
            **kwargs,
            evaluator=self.evaluator,
            use_amp=True,
            **extra_kwargs
        )

        return self.model.evaluate(self.evaluator)
