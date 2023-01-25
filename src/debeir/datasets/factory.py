from pathlib import Path
from typing import Dict, Type, Union

import toml
from debeir.datasets.bioreddit import BioRedditCommentParser, BioRedditSubmissionParser
from debeir.datasets.clinical_trials import ClinicalTrialParser, ClinicalTrialsElasticsearchExecutor, \
    TrialsElasticsearchQuery, TrialsQueryConfig
from debeir.datasets.marco import MarcoElasticsearchExecutor, MarcoQueryConfig
from debeir.datasets.trec_clinical_trials import TrecClincialElasticsearchQuery, TrecClinicalTrialsParser
from debeir.datasets.trec_covid import TrecCovidParser, TrecElasticsearchQuery
from debeir.evaluation.evaluator import Evaluator
from debeir.evaluation.residual_scoring import ResidualEvaluator
from debeir.core.config import Config, ElasticsearchConfig, GenericConfig, MetricsConfig, NIRConfig, SolrConfig, \
    _NIRMasterConfig
from debeir.core.executor import GenericElasticsearchExecutor
from debeir.core.parser import (
    CSVParser, Parser, TSVParser,
)
from debeir.core.query import GenericElasticsearchQuery, Query

str_to_config_cls = {
    "clinical_trials": TrialsQueryConfig,
    "test_trials": TrialsQueryConfig,
    "med-marco": MarcoQueryConfig,
    "generic": MarcoQueryConfig,
}

query_factory = {
    "clinical_trials": TrialsElasticsearchQuery,
    "test_trials": TrialsElasticsearchQuery,
    "generic": GenericElasticsearchQuery,
    "trec_covid": TrecElasticsearchQuery,
    "trec_clinical": TrecClincialElasticsearchQuery,
}

parser_factory = {
    "trec_covid": TrecCovidParser,
    "bioreddit-comment": BioRedditCommentParser,
    "bioreddit-submission": BioRedditSubmissionParser,
    "test_trials": ClinicalTrialParser,
    "med-marco": CSVParser,
    "tsv": TSVParser,
    "trec_clinical": TrecClinicalTrialsParser
}

executor_factory = {
    "clinical": ClinicalTrialsElasticsearchExecutor,
    "med-marco": MarcoElasticsearchExecutor,
    "generic": GenericElasticsearchExecutor,
}

evaluator_factory = {
    "residual": ResidualEvaluator,
    "trec": Evaluator,
}


def get_index_name(config_fp):
    """
    Get the index name from the config without parsing as a TOML

    :param config_fp:
    :return:
    """
    with open(config_fp, "r") as reader:
        for line in reader:
            if line.startswith("index"):
                line = line.replace('"', "")
                return line.split("=")[-1].strip()
    return None


def factory_fn(config_fp, index=None) -> (Query, GenericConfig,
                                          Parser, GenericElasticsearchExecutor, Evaluator):
    """
    Factory method for creating the parsed topics, config object, query object and query executor object

    :param config_fp: Config file path
    :param index: Index to search
    :return:
        Query, Config, Parser, Executor, Evaluator
    """
    config = config_factory(config_fp)
    assert config.index is not None
    query_cls = query_factory[config.query_fn]
    parser = parser_factory[config.parser_fn]
    executor = executor_factory[config.executor_fn]

    return query_cls, config, parser, executor


def config_factory(path: Union[str, Path] = None, config_cls: Type[Config] = None, args_dict: Dict = None):
    """
    Factory method for creating configs

    :param path: Config path
    :param config_cls: Config class to instantiate
    :param args_dict: Arguments to consider
    :return:
        A config object
    """
    if path:
        args_dict = toml.load(path)

    if not config_cls:
        if "config_fn" in args_dict:
            config_cls = str_to_config_cls[args_dict["config_fn"]]
        else:
            raise NotImplementedError()

    return config_cls.from_args(args_dict, config_cls)


def get_nir_config(nir_config, *args, ignore_errors=False, **kwargs):
    main_config = config_factory(nir_config, config_cls=_NIRMasterConfig)
    search_engine_config = None

    supported_search_engines = {"solr": SolrConfig,
                                "elasticsearch": ElasticsearchConfig}

    search_engine_config = None

    if 'engine' in kwargs and kwargs['engine'] in supported_search_engines:
        search_engine = kwargs['engine']
        search_engine_config = config_factory(args_dict=main_config.get_search_engine_settings(search_engine),
                                              config_cls=supported_search_engines[search_engine])

    # for search_engine in supported_search_engines:
    #    if search_engine in kwargs and kwargs[search_engine] and kwargs['engine'] == search_engine:
    #        search_engine_config = config_factory(args_dict=main_config.get_search_engine_settings(search_engine),
    #                                              config_cls=supported_search_engines[search_engine])

    if not ignore_errors and search_engine_config is None:
        raise RuntimeError("Unable to get a search engine configuration.")

    metrics_config = config_factory(args_dict=main_config.get_metrics(), config_cls=MetricsConfig)
    nir_config = config_factory(args_dict=main_config.get_nir_settings(), config_cls=NIRConfig)

    return nir_config, search_engine_config, metrics_config


def apply_nir_config(func):
    """
    Decorator that applies the NIR config settings to the current function
    Replaces arguments and keywords arguments with those found in the config

    :param func:
    :return:
    """

    def parse_nir_config(*args, ignore_errors=False, **kwargs):
        """
        Parses the NIR config for the different setting groups: Search Engine, Metrics and NIR settings
        Applies these settings to the current function
        :param ignore_errors:
        :param args:
        :param kwargs:
        :return:
        """

        nir_config, search_engine_config, metrics_config = get_nir_config(*args,
                                                                          ignore_errors,
                                                                          **kwargs)

        kwargs = nir_config.__update__(
            **search_engine_config.__update__(
                **metrics_config.__update__(**kwargs)
            )
        )

        return func(*args, **kwargs)

    return parse_nir_config
