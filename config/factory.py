from typing import Dict

from config.config import Config, config_factory
from query_builder.elasticsearch.query import GenericQuery, TrialsQuery, Query
from executor.runner import Executor, ClinicalTrialsExecutor, MarcoExecutor, GenericExecutor
from parsers.query_topic_parsers import (
    Parser,
    TrecCovidParser,
    BioRedditCommentParser,
    BioRedditSubmissionParser,
    CDS2021Parser,
    CSVParser,
)


query_factory = {
    "clinical_trials": TrialsQuery,
    "test_trials": TrialsQuery,
    "generic": GenericQuery,
}

parser_factory = {
    "trec_covid": TrecCovidParser,
    "bioreddit-comment": BioRedditCommentParser,
    "bioreddit-submission": BioRedditSubmissionParser,
    "test_trials": CDS2021Parser,
    "med-marco": CSVParser,
}

executor_factory = {
    "clinical": ClinicalTrialsExecutor,
    "med-marco":  MarcoExecutor,
    "generic": GenericExecutor
}


def get_index_name(config_fp):
    with open(config_fp, "r") as reader:
        for line in reader:
            if line.startswith("index"):
                line = line.replace('"', "")
                return line.split("=")[-1].strip()
    return None


def factory_fn(topics_path, config_fp, index=None) -> (Query, Config, Dict, Executor):
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
