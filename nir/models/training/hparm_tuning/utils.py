import abc
import dataclasses
from collections import defaultdict
from enum import Enum
from typing import Sequence, List

import optuna


class Hparam:
    name: str

    @abc.abstractmethod
    def suggest(self, *args, **kwargs):
        raise NotImplementedError()


@dataclasses.dataclass
class HparamFloat(Hparam):
    name: str
    low: float
    high: float
    step: float
    log: bool

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)


@dataclasses.dataclass
class HparamInt(Hparam):
    name: str
    low: int
    high: int
    step: int
    log: bool
    func: str = "suggest_int"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)


@dataclasses.dataclass
class HparamCategorical(Hparam):
    name: str
    choices: Sequence
    func: str = "suggest_categorical"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, self.choices)


@dataclasses.dataclass
class HparamUniform(Hparam):
    name: str
    low: float
    high: float
    func: str = "suggest_uniform"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_uniform(self.name, self.low, self.high)


@dataclasses.dataclass
class HparamLogUniform(Hparam):
    name: str
    low: float
    high: float
    func: str = "suggest_loguniform"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_loguniform(self.name, self.low, self.high)


@dataclasses.dataclass
class HparamDiscreteUniform(Hparam):
    name: str
    low: float
    high: float
    q: float
    func: str = "suggest_discrete_uniform"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_discrete_uniform(self.name, self.low, self.high, self.q)


class HparamTypes(Enum):
    float: HparamFloat
    int: HparamInt
    categorical: HparamCategorical
    uniform: HparamUniform
    loguniform: HparamLogUniform
    discrete_uniform: HparamDiscreteUniform


class TrainerWrapper:
    def __init__(self, trainer_cls, metric):
        self.trainer_cls = trainer_cls
        self.metric = metric

    def fit(self, hparams: Sequence[Hparam], trial: optuna.Trial):
        hparam_values = defaultdict(lambda: 0.0)

        for hparam in hparams:
            hparam_values[hparam.name] = hparam.suggest(trial)


        return hparam_values