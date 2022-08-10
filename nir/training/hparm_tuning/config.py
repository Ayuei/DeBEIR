import dataclasses
from typing import Dict

from training.hparm_tuning.types import HparamTypes
from nir.interfaces.config import Config


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

    def validate(self):
        # Self-validating, errors will be raised if initializations of any object fails.
        return True


    def parse_config_to_py(self):
        """
        Parses configuration file into usable python objects
        """
        hparams = {}

        for hparam, value in self.hparams.items():
            hparam_type = HparamTypes[hparam]

            if "args" in value:  # Of the form {"learning rate": {args: [0.1, 1.0, 0.1]}}
                hparam_obj = hparam_type.__init__(name=hparam, *value["args"])

            else:
                value.pop("type")
                hparam_obj = hparam_type.__init__(name=hparam, **value)

            hparams[hparam] = hparam_obj
        return hparams
