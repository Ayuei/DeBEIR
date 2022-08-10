from collections import defaultdict
from typing import Sequence

import loguru
import optuna

from training.hparm_tuning.config import HparamConfig
from training.hparm_tuning.types import Hparam


class Trainer:
    """
    Wrapper class for a trainer class.

    See Optuna documentation for types!
    """

    def __init__(self, trainer_cls, metric, hparams_config: HparamConfig = None):
        self.trainer_cls = trainer_cls  # Trainer object or method we will initialize
        self.metric = metric

        if hparams_config:
            self.hparams = hparams_config.parse_config_to_py() if hparams_config else None

    def fit(self, trial: optuna.Trial, hparams: Sequence[Hparam] = None):
        """
        Fit the Trainer object using parameters suggested by the optuna library

        :param trial: The optuna trial object
        :param hparams: Optional, pass a dictionary of HparamType[Enum] objects
        :return:
        """

        loguru.logger.info("Fitting the trainer.")

        hparam_values = defaultdict(lambda: 0.0)

        hparams = hparams if hparams else self.hparams

        if hparams is None:
            raise RuntimeError("No hyperparameters were specified")

        for hparam in hparams:
            hparam_values[hparam.name] = hparam.suggest(trial)
            loguru.logger.info(f"Using {hparam_values[hparam.name]} for {hparam.name}.")

        return hparam_values
