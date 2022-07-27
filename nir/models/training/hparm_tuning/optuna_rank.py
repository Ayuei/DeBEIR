from typing import Dict

import torch
import optuna
from functools import partial


def rerank_obj(trainer, hparams: Dict, trial: optuna.Trial):
    train = trainer(hparams, trial)

    metric = train.fit()

    return metric


def run_optuna(trainer, hparams: Dict, n_trials=100, maximize_objective=True):
    """
    Partially initialize the objective function with a trainer and hparams to optimize.

    Optimize using the optuna library.

    :param hparams:
    :param n_trials:
    :param maximize_objective:
    :return:
    """

    assert hasattr(trainer, "fit")

    study = optuna.create_study(direction="maximize" if maximize_objective else "minimize")

    objective = partial(trainer, rerank_obj, hparams)

    study.optimize(objective, n_trials=n_trials, n_jobs=4)
