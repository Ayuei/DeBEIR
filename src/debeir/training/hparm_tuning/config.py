import dataclasses
import json
from typing import Dict

from training.hparm_tuning.types import HparamTypes
from interfaces.config import Config


@dataclasses.dataclass(init=True)
class HparamConfig(Config):
    """
        Hyperparameter configuration file

        Expects a dictionary of hyperparameters

        hparams: Dict
        {
            "learning_rate": {
               "type": float
               "low": 0.1
               "high": 1.0
               "step": 0.1
               # OR
               args: [0.1, 1.0, 0.1]
            },
        }
    """

    hparams: Dict[str, Dict]

    @classmethod
    def from_json(cls, fp) -> "HparamConfig":
        return HparamConfig(json.load(open(fp)))

    def validate(self):
        # Self-validating, errors will be raised if initializations of any object fails.
        return True

    def parse_config_to_py(self):
        """
        Parses configuration file into usable python objects
        """
        hparams = {}

        for hparam, value in self.hparams.items():
            #if "args" in value:  # Of the form {"learning rate": {args: [0.1, 1.0, 0.1]}}
            #    hparam_obj = hparam_type(name=hparam, *value["args"])
            if isinstance(value, Dict) and "type" in value:
                hparam_type = HparamTypes[value['type']]
                value.pop("type")
                hparam_obj = hparam_type(name=hparam, **value)
            else:
                hparam_obj = value

            hparams[hparam] = hparam_obj

        return hparams
