import asyncio
import importlib
import os
import sys
import toml

import pytest


@pytest.fixture()
def config_file_dict(tmp_path_factory):
    sys.path.insert(0, os.path.dirname("../main.py"))

    config = {
        "query_type": "embedding",
        "automatic": True,
        "encoder_fp": "sentence-transformers/distilbert-base-nli-mean-tokens",
        "index": "test",
        "cosine_ceiling": 5.893593,

        "config_fn": "generic",
        "query_fn": "generic",
        "parser_fn": "generic",
        "executor_fn": "generic",

        "qrels": "./assets/qrels2021.txt",
        "topics_path": "./assets/topics2022.xml"
    }

    fn = tmp_path_factory.mktemp("test_config") / "temp_config.toml"
    toml.dump(config, open(fn, "w+"))

    return fn, config


def test_config_load(config_file_dict):
    from nir.interfaces import config

    config_file, config_dict = config_file_dict
    c = config.Config.from_toml(config_file, config.GenericConfig)

    for key in config_dict:
        assert getattr(c, key) == config_dict[key]

    c_args = config.Config.from_args(config_dict, config.GenericConfig)

    for key in config_dict:
        assert getattr(c_args, key) == config_dict[key]


def test_bm25_change():
    pass


def test_evaluation():
    pass


def test_config_load():
    pass


def test_embedding_queries():
    importlib.import_module("../main.py")

    asyncio.run(
        main.main(
            topics="../assets/topics-rnd5.xml",
            configs=["../configs/trec_covid/embedding.toml"],
            nir_config="../configs/nir.toml",
            elasticsearch=True)
    )
