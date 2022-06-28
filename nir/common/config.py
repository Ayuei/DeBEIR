import abc
import dataclasses
from abc import ABC
from dataclasses import dataclass
from typing import List, MutableMapping, Dict

import toml

from rankers.transformer_encoder import Encoder


class Config:
    """
    Config Interface with creation class methods
    """
    def __update__(self, **kwargs):
        attrs = vars(self)
        kwargs.update(attrs)

        return kwargs

    @classmethod
    def from_toml(cls, fp: str, field_class, *args, **kwargs) -> 'Config':
        """
        Instantiates a Config object from a toml file

        :param fp: File path of the Config TOML file
        :param field_class: Class of the Config object to be instantiated
        :param args: Arguments to be passed to Config
        :param kwargs: Keyword arguments to be passed
        :return:
            A instantiated and validated Config object.
        """
        args_dict = toml.load(fp)

        return cls.from_args(args_dict, field_class, *args, **kwargs)

    @classmethod
    def from_args(cls, args_dict: MutableMapping, field_class, *args, **kwargs):
        """
        Instantiates a Config object from arguments

        :param args_dict:
        :param field_class:
        :param args:
        :param kwargs:
        :return:
        """
        field_names = set(f.name for f in dataclasses.fields(field_class))
        obj = field_class(**{k: v for k, v in args_dict.items() if k in field_names})
        if hasattr(obj, 'encoder_fp') and obj.encoder_fp:
            obj.encoder = Encoder(obj.encoder_fp, obj.encoder_normalize)

        obj.validate()

        return obj

    @classmethod
    def from_dict(cls, data_class, **kwargs):
        """
        Instantiates a Config object from a dictionary

        :param data_class:
        :param kwargs:
        :return:
        """
        if "encoder_fp" in kwargs and kwargs["encoder_fp"]:
            kwargs["encoder"] = Encoder(kwargs["encoder_fp"])

        field_names = set(f.name for f in dataclasses.fields(data_class))
        obj = data_class(**{k: v for k, v in kwargs.items() if k in field_names})
        obj.validate()

        return obj

    @abc.abstractmethod
    def validate(self):
        """
        Validates if the config is correct.
        Must be implemented by inherited classes.
        """
        pass


@dataclass(init=True, unsafe_hash=True)
class GenericConfig(Config, ABC):
    """
    Generic NIR Configuration file for which all configs will inherit
    """
    query_type: str
    index: str = None
    encoder_normalize: bool = True
    ablations: bool = False
    norm_weight: float = None
    automatic: bool = None
    encoder: Encoder = None
    encoder_fp: str = None
    query_weights: List[float] = None
    cosine_weights: List[float] = None
    evaluate: bool = False
    qrels: str = None
    config_fn: str = None
    query_fn: str = None
    parser_fn: str = None
    executor_fn: str = None


@dataclass(init=True)
class _NIRMasterConfig(Config):
    """
    Base NIR Master config: nir.toml
    """
    metrics: Dict
    search: Dict
    nir: Dict

    def get_metrics(self, key='common'):
        return self.metrics[key]

    def get_search_engine_settings(self, key='elasticsearch'):
        return self.search['engines'][key]

    def get_nir_settings(self, key='default_settings'):
        return self.nir[key]

    def validate(self):
        return True


@dataclass(init=True)
class ElasticsearchConfig(Config):
    """
    Basic Elasticsearch configuration file settings from the master nir.toml file
    """
    protocol: str
    ip: str
    port: str
    timeout: int

    def validate(self):
        """
        Checks if Elasticsearch URL is correct
        """
        assert self.protocol in ['http', 'https']
        assert self.port.isdigit()


@dataclass(init=True)
class SolrConfig(ElasticsearchConfig):
    """
    Basic Solr configuration file settings from the master nir.toml file
    """
    pass


@dataclass(init=True)
class MetricsConfig(Config):
    """
    Basic Metrics configuration file settings from the master nir.toml file
    """
    metrics: List[str]

    def validate(self):
        """
        Checks if each Metrics is usable by evaluator classes
        """
        for metric in self.metrics:
            assert "@" in metric

            metric, depth = metric.split("@")

            assert metric.isalpha()
            assert depth.isdigit()


@dataclass(init=True)
class NIRConfig(Config):
    """
    Basic NIR configuration file settings from the master nir.toml file
    """
    norm_weight: str
    overwrite_output_if_exists: bool
    evaluate: bool
    return_size: int
    output_directory: str

    def validate(self):
        return True


def apply_config(func):
    """
    Configuration decorator.

    :param func: Decorated function
    :return:
    """
    def use_config(self, *args, **kwargs):
        """
        Replaces keywords and args passed to the function with ones from self.config.

        :param self:
        :param args: To be updated
        :param kwargs: To be updated
        :return:
        """
        if self.config is not None:
            kwargs = self.config.__update__(**kwargs)

        return func(self, *args, **kwargs)

    return use_config
