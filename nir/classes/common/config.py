import abc
import dataclasses
from dataclasses import dataclass
from typing import List, MutableMapping, Dict

import toml

from nir.utils.embeddings import Encoder


class Config:
    def __update__(self, **kwargs):
        attrs = vars(self)
        kwargs.update(attrs)

        return kwargs

    @classmethod
    def from_toml(cls, fp: str, field_class, *args, **kwargs) -> 'Config':
        args_dict = toml.load(fp)

        return cls.from_args(args_dict, field_class, *args, **kwargs)

    @classmethod
    def from_args(cls, args_dict: MutableMapping, field_class, *args, **kwargs):
        field_names = set(f.name for f in dataclasses.fields(field_class))
        obj = field_class(**{k: v for k, v in args_dict.items() if k in field_names})
        if hasattr(obj, 'encoder_fp') and obj.encoder_fp:
            obj.encoder = Encoder(obj.encoder_fp, obj.encoder_normalize)

        obj.validate()

        return obj

    @classmethod
    def from_dict(cls, data_class, **kwargs):
        if "encoder_fp" in kwargs and kwargs["encoder_fp"]:
            kwargs["encoder"] = Encoder(kwargs["encoder_fp"])

        field_names = set(f.name for f in dataclasses.fields(data_class))
        obj = data_class(**{k: v for k, v in kwargs.items() if k in field_names})
        obj.validate()

        return obj

    @abc.abstractmethod
    def validate(self):
        pass


@dataclass(init=True, unsafe_hash=True)
class GenericConfig(Config):
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

    @abc.abstractmethod
    def validate(self):
        pass


@dataclass(init=True)
class _NIRMasterConfig(Config):
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
    protocol: str
    ip: str
    port: str

    def validate(self):
        assert self.protocol in ['http', 'https']
        assert self.port.isdigit()


@dataclass(init=True)
class SolrConfig(ElasticsearchConfig):
    pass


@dataclass(init=True)
class MetricsConfig(Config):
    metrics: List[str]

    def validate(self):
        for metric in self.metrics:
            assert "@" in metric

            metric, depth = metric.split("@")

            assert metric.isalpha()
            assert depth.isdigit()


@dataclass(init=True)
class NIRConfig(Config):
    norm_weight: str
    remove: bool
    evaluate: bool
    return_size: int
    output_directory: str

    def validate(self):
        return True


def apply_config(func):
    def use_config(self, *args, **kwargs):
        if self.config is not None:
            kwargs = self.config.__update__(**kwargs)

        return func(self, *args, **kwargs)

    return use_config


