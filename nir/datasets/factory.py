from typing import Dict, Type

import toml

from nir.interfaces.config import GenericConfig, _NIRMasterConfig, SolrConfig, ElasticsearchConfig, MetricsConfig, \
    NIRConfig, Config
from nir.interfaces.query import GenericElasticsearchQuery
from nir.datasets.clinical_trials import TrialsElasticsearchQuery
from nir.datasets.trec_covid import TrecElasticsearchQuery
from nir.engines.elasticsearch.executor import ElasticsearchExecutor
from nir.datasets.clinical_trials import (
    ClinicalTrialsExecutor,
    ClinicalTrialParser,
    TrialsQueryConfig,
)
from nir.datasets.marco import MarcoExecutor, MarcoQueryConfig
from nir.interfaces.executor import GenericExecutor
from nir.interfaces.parser import (
    CSVParser,
)
from nir.datasets.bioreddit import BioRedditSubmissionParser, BioRedditCommentParser
from nir.datasets.trec_covid import TrecCovidParser

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
}

parser_factory = {
    "trec_covid": TrecCovidParser,
    "bioreddit-comment": BioRedditCommentParser,
    "bioreddit-submission": BioRedditSubmissionParser,
    "test_trials": ClinicalTrialParser,
    "med-marco": CSVParser,
}

executor_factory = {
    "clinical": ClinicalTrialsExecutor,
    "med-marco": MarcoExecutor,
    "generic": GenericExecutor,
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


def factory_fn(
    topics_path, config_fp, index=None
) -> (GenericElasticsearchQuery, GenericConfig, Dict, ElasticsearchExecutor):
    """
    Factory method for creating the parsed topics, config object, query object and query executor object

    :param topics_path: Path to topics
    :param config_fp: Config file path
    :param index: Index to search
    :return:
        Query, Config, Topics, Topics, Executor
    """
    if index is None:
        index = get_index_name(config_fp)
        assert (
            index is not None
        ), "Index must be provided in the config file or as an an argument"

    config = config_factory(config_fp)
    query_fct = query_factory[config.query_fn]
    parser = parser_factory[config.parser_fn]
    executor = executor_factory[config.executor_fn]
    query_type = config.query_type

    topics = parser.get_topics(open(topics_path))
    query = query_fct(topics=topics, query_type=query_type, config=config)

    return query, config, topics, executor


def config_factory(path: str = None, config_cls: Type[Config] = None, args_dict: Dict = None):
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


def apply_nir_config(func):
    """
    Decorator that applies the NIR config settings to the current function
    Replaces arguments and keywords arguments with those found in the config

    :param func:
    :return:
    """
    def parse_nir_config(*args, **kwargs):
        """
        Parses the NIR config for the different setting groups: Search Engine, Metrics and NIR settings
        Applies these settings to the current function
        :param args:
        :param kwargs:
        :return:
        """
        main_config = config_factory(kwargs['nir_config'], config_cls=_NIRMasterConfig)
        search_engine_config = None

        supported_search_engines = {"solr": SolrConfig,
                                    "elasticsearch": ElasticsearchConfig}

        for search_engine in supported_search_engines:
            if search_engine in kwargs and kwargs[search_engine] and kwargs['engine'] == search_engine:
                search_engine_config = config_factory(args_dict=main_config.get_search_engine_settings(search_engine),
                                                      config_cls=supported_search_engines[search_engine])

        if search_engine_config is None:
            raise RuntimeError("Unable to get a search engine configuration.")

        metrics_config = config_factory(args_dict=main_config.get_metrics(), config_cls=MetricsConfig)
        nir_config = config_factory(args_dict=main_config.get_nir_settings(), config_cls=NIRConfig)

        kwargs = nir_config.__update__(
            **search_engine_config.__update__(
                **metrics_config.__update__(**kwargs)
            )
        )

        return func(*args, **kwargs)

    return parse_nir_config
