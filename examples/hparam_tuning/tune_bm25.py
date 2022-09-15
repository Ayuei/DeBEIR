import asyncio
import time
from functools import partial

import sys

import shutup
from wandb import wandb
from loguru import logger

sys.path.append("/home/vin/Projects/nir/")

from nir.engines.elasticsearch.change_bm25 import change_bm25_params
import joblib
import optuna
from optuna.integration import WeightsAndBiasesCallback

from nir.engines.client import Client
from nir.data_sets.factory import factory_fn
from main import run_config_es
from nir.training.hparm_tuning.optuna_rank import print_optuna_stats


def parse_arguments(
        topics,
        configs,
        **kwargs
):
    """
    Main function loop. Executes the passed config files and executes them all asynchronously.

    :param topics: Input topics (file path)
    :param configs: File paths to configs
    :param debug: Debug flag
    :param kwargs: Additional arguments from cmd args to pass to run configs
    """

    client = Client()
    config_fp = configs[0]

    query_cls, config, parsed_topics, executor_cls = factory_fn(topics, config_fp)

    new_kwargs = {"query_cls": query_cls, "config": config, "parsed_topics": parsed_topics,
                  "executor_cls": executor_cls,
                  "config_fp": config_fp, "client": client}

    new_kwargs.update(kwargs)

    return new_kwargs


def recall_objective(kwargs, trial: optuna.Trial):
    k1 = trial.suggest_float("k", low=0.01, high=4.0)
    b = trial.suggest_float("b", low=0.01, high=1.0)

    change_bm25_params(kwargs['config'].index, k1, b)

    loop = asyncio.get_event_loop()

    results = loop.run_until_complete(run_config_es(kwargs['parsed_topics'], kwargs['config'], kwargs['config_fp'],
                                        kwargs['executor_cls'], kwargs['query_cls'],
                                        client=kwargs['client'], engine='elasticsearch',
                                        nir_config="/home/vin/Projects/nir/configs/nir.toml",
                                        filter_ids=None,
                                        elasticsearch=True,
                                        debug=False))

    wandb.log(results)

    return float(results['num_rel_ret']) / float(results['num_rel'])


def run_optuna_with_wandb(kwargs, n_trials=1000, n_jobs=1, maximize_objective=True, save_study_path="."):
    wandb_kwargs = {"project": "bm25_tuning"}

    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction="maximize" if maximize_objective else "minimize")
    obj = partial(recall_objective, kwargs)

    try:
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, callbacks=[wandbc])
    except:
        pass
    finally:
        joblib.dump(study, save_study_path + ".pkl")

    return study


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    kwargs = parse_arguments("/home/vin/Projects/nir/assets/topics2021.xml",
                             ["/home/vin/Projects/nir/configs/trec2022/baseline.toml"],
                             filter_ids="/home/vin/Projects/nir/filter_ids.json")

    study = run_optuna_with_wandb(kwargs)

    print_optuna_stats(study)
