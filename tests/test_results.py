import os
import sys

import pytest
import toml

from src.debeir.engines.client import Client
from src.debeir.data_sets.factory import query_factory, parser_factory, executor_factory, config_factory
from src.debeir.interfaces import config
from src.debeir.interfaces.config import _NIRMasterConfig
from src.debeir.interfaces.results import Results


@pytest.fixture(scope="session")
def config_file_dict(tmp_path_factory):
    sys.path.insert(0, os.path.dirname("../main.py"))

    config = {
        "query_type": "embedding",
        "automatic": True,
        "encoder_fp": "sentence-transformers/distilbert-base-nli-mean-tokens",
        "index": "test",
        "norm_weight": 10,

        "config_fn": "generic",
        "query_fn": "generic",
        "parser_fn": "tsv",
        "executor_fn": "generic",

        "qrels": "test_set/qrels.tsv",
        "topics_path": "test_set/queries.tsv",
    }

    fn = tmp_path_factory.mktemp("test_config") / "temp_config.toml"
    toml.dump(config, open(fn, "w+"))

    return fn, config


@pytest.fixture(scope="session")
def nir_config_dict(tmp_path_factory):
    nir_str = """
        [metrics]
            [metrics.common]
            metrics=["ndcg@10", "ndcg@20", "ndcg@100",
                     "rprec@1000",
                     "p@1", "p@5", "p@10", "p@15", "p@20",
                     "bpref@1000",
                     "recall@1000",
                     "rprec@1000",
                     "r@1000"]
        
        [search.engines]
            [search.engines.elasticsearch]
            protocol = "http"
            ip = "localhost"
            port = "9200"
            timeout = 600
        
            [search.engines.solr]
            ip = "127.0.0.1"
            port = "9200"
        
        
        [nir]
            [nir.default_settings]
            norm_weight = "2.15"
            evaluate = false
            return_size = 5 
            output_directory = "./output/test/"
    """

    fn = tmp_path_factory.mktemp("test_config") / "temp_nir_config.toml"

    nir_toml = toml.loads(nir_str)
    toml.dump(nir_toml, open(fn, 'w+t'))

    return fn, nir_toml


@pytest.mark.asyncio
async def test_iteration_topic_num(config_file_dict, nir_config_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])

    parser_cls = parser_factory[c.parser_fn]
    topics = parser_cls._get_topics(c.topics_path)
    query_cls = query_factory[c.query_fn]
    engine_cls = executor_factory[c.executor_fn]

    master_config = config_factory(nir_config_dict[0], _NIRMasterConfig)
    engine_config = master_config.get_search_engine_settings(return_as_instance=True)
    nir_settings = master_config.get_nir_settings(return_as_instance=True)

    client = Client.build_from_config("elasticsearch", engine_config)

    query_obj = query_cls(topics=topics, config=c)
    engine = engine_cls.build_from_config(topics, query_obj, client.es_client, c,
                                          nir_settings)

    results = await engine.run_all_queries(return_results=True, cosine_offset=100, return_size=5)

    r = Results(results, query_obj, "elasticsearch")

    for (topic_num, res) in results:
        r_itr_topic = iter(r(topic_num=topic_num))
        for rank, result in enumerate(res["hits"]["hits"], start=1):
            doc_id = query_cls.get_id_mapping(result["_source"])
            doc = next(r_itr_topic)

            assert doc.doc_id == doc_id
            assert doc.topic_num == topic_num
            assert doc.scores['rank'] == rank
            assert doc.facets['Text'] == result['_source']['Text']

@pytest.mark.asyncio
async def test_iteration_no_topic_num(config_file_dict, nir_config_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])

    parser_cls = parser_factory[c.parser_fn]
    topics = parser_cls._get_topics(c.topics_path)
    query_cls = query_factory[c.query_fn]
    engine_cls = executor_factory[c.executor_fn]

    master_config = config_factory(nir_config_dict[0], _NIRMasterConfig)
    engine_config = master_config.get_search_engine_settings(return_as_instance=True)
    nir_settings = master_config.get_nir_settings(return_as_instance=True)

    client = Client.build_from_config("elasticsearch", engine_config)

    query_obj = query_cls(topics=topics, config=c)
    engine = engine_cls.build_from_config(topics, query_obj, client.es_client, c,
                                          nir_settings)

    results = await engine.run_all_queries(return_results=True, cosine_offset=100, return_size=5)


    r = Results(results, query_obj, "elasticsearch")

    temp = {}

    for topic_num, res in results:
        temp[topic_num] = res

    results = temp
    r_itr_topic = iter(r)

    for topic_num in r.get_topic_ids():
        res = results[topic_num]

        for rank, result in enumerate(res["hits"]["hits"], start=1):
            doc_id = query_cls.get_id_mapping(result["_source"])
            doc = next(r_itr_topic)

            assert doc.doc_id == doc_id
            assert doc.topic_num == topic_num
            assert doc.scores['rank'] == rank
            assert doc.facets['Text'] == result['_source']['Text']


@pytest.mark.asyncio
async def test_iteration_dict(config_file_dict, nir_config_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])

    parser_cls = parser_factory[c.parser_fn]
    topics = parser_cls._get_topics(c.topics_path)
    query_cls = query_factory[c.query_fn]
    engine_cls = executor_factory[c.executor_fn]

    master_config = config_factory(nir_config_dict[0], _NIRMasterConfig)
    engine_config = master_config.get_search_engine_settings(return_as_instance=True)
    nir_settings = master_config.get_nir_settings(return_as_instance=True)

    client = Client.build_from_config("elasticsearch", engine_config)

    query_obj = query_cls(topics=topics, config=c)
    engine = engine_cls.build_from_config(topics, query_obj, client.es_client, c,
                                          nir_settings)

    results = await engine.run_all_queries(return_results=True, cosine_offset=100, return_size=5)
    r = Results(results, query_obj, "elasticsearch")

    temp = {}

    for topic_num, res in results:
        temp[topic_num] = res

    results = temp

    for topic_num in r.get_topic_ids():
        res = results[topic_num]
        result_set = r[topic_num]

        for rank, result in enumerate(res["hits"]["hits"]):
            doc_id = query_cls.get_id_mapping(result["_source"])
            doc = result_set[rank]

            assert doc.doc_id == doc_id
            assert doc.topic_num == topic_num
            assert doc.scores['rank'] == rank + 1
            assert doc.facets['Text'] == result['_source']['Text']
