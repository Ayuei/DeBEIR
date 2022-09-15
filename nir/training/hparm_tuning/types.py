import abc
import dataclasses
from enum import Enum
from typing import Sequence, Dict

import optuna


class Hparam:
    name: str

    @abc.abstractmethod
    def suggest(self, *args, **kwargs):
        raise NotImplementedError()


@dataclasses.dataclass(init=True)
class HparamFloat(Hparam):
    name: str
    low: float
    high: float
    log: bool = False
    step: float = None

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)


@dataclasses.dataclass(init=True)
class HparamInt(Hparam):
    name: str
    low: int
    high: int
    log: bool = False
    step: int = 1

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)


@dataclasses.dataclass(init=True)
class HparamCategorical(Hparam):
    name: str
    choices: Sequence
    func: str = "suggest_categorical"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, self.choices)


@dataclasses.dataclass(init=True)
class HparamUniform(Hparam):
    name: str
    low: float
    high: float
    func: str = "suggest_uniform"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_uniform(self.name, self.low, self.high)


@dataclasses.dataclass(init=True)
class HparamLogUniform(Hparam):
    name: str
    low: float
    high: float
    func: str = "suggest_loguniform"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_loguniform(self.name, self.low, self.high)


@dataclasses.dataclass(init=True)
class HparamDiscreteUniform(Hparam):
    name: str
    low: float
    high: float
    q: float
    func: str = "suggest_discrete_uniform"

    def suggest(self, trial: optuna.Trial):
        return trial.suggest_discrete_uniform(self.name, self.low, self.high, self.q)


HparamTypes = {
    "float": HparamFloat,
    "int": HparamInt,
    "categorical": HparamCategorical,
    "uniform": HparamUniform,
    "loguniform": HparamLogUniform,
    "discrete_uniform": HparamDiscreteUniform
}
