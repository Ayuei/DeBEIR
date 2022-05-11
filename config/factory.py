from config.config import MarcoQueryConfig, TrialsQueryConfig, Config
from query_builder.elasticsearch.query import GenericQuery, TrialsQuery, Query
from executor.runner import Executor, ClinicalTrialsExecutor, MarcoExecutor
from parsers.query_topic_parsers import (
    Parser,
    TrecCovidParser,
    BioRedditCommentParser,
    BioRedditSubmissionParser,
    CDS2021Parser,
    CSVParser,
)

config_factory = {
    "clinical_trials": TrialsQueryConfig,
    "test_trials": TrialsQueryConfig,
    "med-marco": MarcoQueryConfig,
}

query_factory = {
    "clinical_trials": TrialsQuery,
    "test_trials": TrialsQuery,
    "med-marco": GenericQuery,
}

parser_factory = {
    "trec_covid": TrecCovidParser,
    "bioreddit-comment": BioRedditCommentParser,
    "bioreddit-submission": BioRedditSubmissionParser,
    "test_trials": CDS2021Parser,
    "med-marco": CSVParser,
}

executor_factory = {
    "trec_covid": ClinicalTrialsExecutor,
    "med-marco":  MarcoExecutor,
}


def get_index_name(config_fp):
    with open(config_fp, "r") as reader:
        for line in reader:
            if line.startswith("index"):
                line = line.replace('"', "")
                return line.split("=")[-1].strip()
    return None


def factory_fn(topics, config_fp, index=None) -> (Query, Config, Parser, Executor):
    if index is None:
        index = get_index_name(config_fp)
        assert (
            index is not None
        ), "Index must be provided in the config file or as an an argument"

    config_fct = config_factory[index]
    query_fct = query_factory[index]
    config = config_fct.from_toml(config_fp)
    parser = parser_factory[index]
    executor = executor_factory[index]
    query_type = config.query_type

    query = query_fct(topics=topics, query_type=query_type, config=config)

    return query, config, parser, executor
