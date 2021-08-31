import dataclasses
import os
from dataclasses import dataclass
from utils.embeddings import Encoder
import toml
from typing import List


@dataclass(init=True)
class Config:
    def validate(self):
        raise NotImplementedError()

    def __update__(self, **kwargs):
        attrs = vars(self)

        # Overwrite with config value if it exists
        #for attr in attrs:
        #    if attr in kwargs:
        #        kwargs[attr] = attrs[attr]

        kwargs.update(attrs)

        return kwargs

    @classmethod
    def from_toml(cls, fp: str) -> 'Config':
        raise NotImplementedError


@dataclass(init=True)
class TrialsQueryConfig(Config):
    query_type: str
    query_field_usage: str = None
    embed_field_usage: str = None
    fields: List[str] = None
    ablations: bool = False
    norm_weight: float = None
    automatic: bool = None
    query_weights: List[float] = None
    cosine_weights: List[float] = None
    encoder: Encoder = None
    encoder_fp: str = None
    index: str = None
    encoder_normalize: bool = True

    def validate(self):
        if self.query_type == "embedding":
            assert self.query_field_usage and self.embed_field_usage, "Must have both field usages" \
                                                                            " if embedding query"
            assert self.encoder_fp and self.encoder, "Must provide encoder path for embedding model"
            assert self.norm_weight is not None or self.automatic is not None, "Norm weight be specified or be automatic"

        assert self.query_field_usage is not None or self.fields is not None, "Must have a query field"
        assert self.query_type in ["ablation", "query", "query_best", "embedding"], "Check your query type"

    @classmethod
    def from_toml(cls, fp: str) -> 'TrialsQueryConfig':
        args_dict = toml.load(fp)

        field_names = set(f.name for f in dataclasses.fields(TrialsQueryConfig))
        obj = cls(**{k: v for k, v in args_dict.items() if k in field_names})

        if obj.encoder_fp:
            obj.encoder = Encoder(obj.encoder_fp, obj.encoder_normalize)

        obj.validate()

        return obj

    @classmethod
    def from_dict(cls, kwargs) -> 'TrialsQueryConfig':

        if 'encoder_fp' in kwargs and kwargs ['encoder_fp']:
            kwargs['encoder'] = Encoder(kwargs['encoder_fp'])

        field_names = set(f.name for f in dataclasses.fields(TrialsQueryConfig))
        obj = cls(**{k: v for k, v in kwargs.items() if k in field_names})
        obj.validate()

        return obj


@dataclass(init=True)
class MarcoQueryConfig(Config):
    query_type: str
    ablations: bool = False
    norm_weight: float = None
    automatic: bool = None
    query_weights: List[float] = None
    cosine_weights: List[float] = None
    encoder: Encoder = None
    encoder_fp: str = None
    index: str = None
    encoder_normalize: bool = True

    def validate(self):
        if self.query_type == "embedding":
            assert self.encoder_fp and self.encoder, "Must provide encoder path for embedding model"
            assert self.norm_weight is not None or self.automatic is not None, "Norm weight be " \
                                                                               "specified or be automatic"

    @classmethod
    def from_toml(cls, fp: str) -> 'MarcoQueryConfig':
        args_dict = toml.load(fp)

        field_names = set(f.name for f in dataclasses.fields(MarcoQueryConfig))
        obj = cls(**{k: v for k, v in args_dict.items() if k in field_names})
        if obj.encoder_fp:
            obj.encoder = Encoder(obj.encoder_fp, obj.encoder_normalize)

        obj.validate()

        return obj

    @classmethod
    def from_dict(cls, kwargs) -> 'MarcoQueryConfig':
        if 'encoder_fp' in kwargs and kwargs ['encoder_fp']:
            kwargs['encoder'] = Encoder(kwargs['encoder_fp'])

        field_names = set(f.name for f in dataclasses.fields(MarcoQueryConfig))
        obj = cls(**{k: v for k, v in kwargs.items() if k in field_names})
        obj.validate()

        return obj


def apply_config(func):
    def use_config(self, *args, **kwargs):
        if self.config is not None:
            kwargs = self.config.__update__(**kwargs)

        return func(self, *args, **kwargs)

    return use_config
