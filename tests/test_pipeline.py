import os
import sys

import loguru
import pytest
import requests
import toml

from engines.client import Client
from interfaces.pipeline import NIRPipeline
from nir.data_sets.factory import query_factory, parser_factory, executor_factory, config_factory
from nir.engines.elasticsearch.change_bm25 import change_bm25_params
from nir.interfaces import config
from nir.interfaces.config import _NIRMasterConfig


@pytest.fixture(scope="session")
def config_file_dict(tmp_path_factory):
    sys.path.insert(0, os.path.dirname("../main.py"))

    config = {
        "query_type": "embedding",
        "automatic": True,
        "encoder_fp": "sentence-transformers/distilbert-base-nli-mean-tokens",
        "index": "test",
        "norm_weight": 50,

        "config_fn": "generic",
        "query_fn": "generic",
        "parser_fn": "tsv",
        "executor_fn": "generic",

        "qrels": "test_set/qrels_dev.tsv",
        "topics_path": "test_set/queries.tsv"
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
            ip = "127.0.0.1"
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


def check_config_instance_has_right_attributes(config, config_dict):
    for key in config_dict:
        assert getattr(config, key) == config_dict[key]


def test_config_load(config_file_dict):
    from nir.interfaces import config

    config_file, config_dict = config_file_dict
    c = config_factory(config_file, config.GenericConfig)

    loguru.logger.info("Loaded Config")

    check_config_instance_has_right_attributes(c, config_dict)

    c = config_factory(args_dict=config_dict, config_cls=config.GenericConfig)

    check_config_instance_has_right_attributes(c, config_dict)


def test_nir_conifg_load(nir_config_dict):
    nir_file, nir_dict = nir_config_dict

    master_config = config_factory(nir_file, _NIRMasterConfig)

    metrics_config = master_config.get_metrics(return_as_instance=True)
    engine_config = master_config.get_search_engine_settings(return_as_instance=True)
    nir_settings = master_config.get_nir_settings(return_as_instance=True)

    check_config_instance_has_right_attributes(metrics_config, master_config.get_metrics())

    # Sanity check
    assert metrics_config.metrics == ["ndcg@10", "ndcg@20", "ndcg@100", "rprec@1000", "p@1", "p@5", "p@10",
                                      "p@15", "p@20", "bpref@1000", "recall@1000", "rprec@1000", "r@1000"]

    check_config_instance_has_right_attributes(engine_config, master_config.get_search_engine_settings())
    check_config_instance_has_right_attributes(nir_settings, master_config.get_nir_settings())


@pytest.mark.xfail(reason="Sanity check, Should fail")
def test_config_load_fail(config_file_dict):
    config_file, config_dict = config_file_dict
    c = config.Config.from_toml(config_file, config.GenericConfig)

    loguru.logger.info("Loaded Config")
    key_doesnt_exist = "missing_key"

    assert getattr(c, key_doesnt_exist) == config_dict[key_doesnt_exist]


def test_bm25_change(config_file_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0], config.GenericConfig)
    k1, b = 0.1, 0.1

    change_bm25_params(index=c.index, k1=k1, b=b)

    resp = requests.get(f"http://localhost:9200/{c.index}/_settings?pretty",
                        timeout=10)

    expected_resp = {
        'k1': str(k1),
        'b': str(b),
        'type': 'BM25'
    }

    similarity_resp = resp.json()[c.index]['settings']['index']['similarity']['default']

    for key in similarity_resp:
        assert similarity_resp[key] == expected_resp[key]

    k1, b = 1.2, 0.75
    change_bm25_params(index=c.index, k1=1.2, b=0.75)

    resp = requests.get(f"http://localhost:9200/{c.index}/_settings?pretty",
                        timeout=10)

    similarity_resp = resp.json()[c.index]['settings']['index']['similarity']['default']

    expected_resp = {
        'k1': str(k1),
        'b': str(b),
        'type': 'BM25'
    }

    for key in similarity_resp:
        assert similarity_resp[key] == expected_resp[key]


def test_parser_cls_method(config_file_dict):
    c = config.Config.from_toml(config_file_dict[0], config.GenericConfig)

    parser = parser_factory[c.parser_fn]

    parsed_topics = parser._get_topics(c.topics_path)

    for topic_num, topic_text in parsed_topics.items():
        assert isinstance(topic_num, int | float) or topic_num.isdigit()
        assert len(topic_text['text']) > 0
        assert isinstance(topic_text['text'], str)

    assert '118' in parsed_topics
    assert '1185869' in parsed_topics
    assert len(parsed_topics) == 3048


def test_parser_instance_method(config_file_dict):
    c = config.Config.from_toml(config_file_dict[0], config.GenericConfig)

    parser = parser_factory[c.parser_fn]
    parser.parse_fields = ["invalid_field", "invalid_field", "invalid_field"]
    parser = parser(parse_fields=["id", "text"])

    parsed_topics = parser.get_topics(c.topics_path)

    for topic_num, topic in parsed_topics.items():
        assert isinstance(topic_num, int | float) or topic_num.isdigit()
        assert len(topic['text']) > 0
        assert topic['id'] == topic_num
        assert isinstance(topic['text'], str)

    assert '118' in parsed_topics
    assert '1185869' in parsed_topics
    assert len(parsed_topics) == 3048


@pytest.mark.asyncio
async def test_query_generation(config_file_dict, nir_config_dict):
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

    for topic_num, topic in topics.items():
        query = query_obj.generate_query(topic_num=topic_num, query_type="query")

        assert query['query']['bool']['should'][0]['match']['Text']['query'] == topic['text']

        q, res = await engine.execute_query(query, return_size=5,
                                            return_id_only=False)

        q_documents = [doc['_source'] for doc in res['hits']['hits']]
        assert len(q_documents) == 5
        assert q == query

        t_num, res = await engine.execute_query(query=None, topic_num=topic_num, return_size=5,
                                                return_id_only=False,
                                                query_type="query")

        t_documents = [doc['_source'] for doc in res['hits']['hits']]

        for doc1, doc2 in zip(q_documents, t_documents):
            assert doc1['Id'] == doc2['Id']
            assert doc1['Text'] == doc2['Text']

        assert topic_num == t_num

        await client.close()


def test_query_encoder(config_file_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])

    parser_cls = parser_factory[c.parser_fn]
    topics = parser_cls._get_topics(c.topics_path)
    query_cls = query_factory[c.query_fn]
    query_obj = query_cls(topics=topics, config=c)

    for topic_num, topic in topics.items():
        text = topic['text']
        encoded_text = c.encoder.encode(c.encoder, topic=text)

        embedding_query = query_obj.generate_query_embedding(topic_num=topic_num)
        encoded_query = embedding_query['query']['script_score']['script']['params']['text_eb']

        assert encoded_text == encoded_query


def test_query_encoder_cache(config_file_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])

    parser_cls = parser_factory[c.parser_fn]
    topics = parser_cls._get_topics(c.topics_path)
    query_cls = query_factory[c.query_fn]
    query_obj = query_cls(topics=topics, config=c)

    import time

    start = time.time()

    for topic_num, topic in topics.items():
        text = topic['text']
        encoded_text = c.encoder.encode(c.encoder, topic=text)

        embedding_query = query_obj.generate_query_embedding(topic_num=topic_num)
        encoded_query = embedding_query['query']['script_score']['script']['params']['text_eb']

        assert encoded_text == encoded_query

    end = time.time()
    cache_time = end - start

    start = time.time()

    for topic_num, topic in topics.items():
        text = topic['text']
        encoded_text = c.encoder.encode(c.encoder, topic=text, disable_cache=True)

        embedding_query = query_obj.generate_query_embedding(topic_num=topic_num)
        encoded_query = embedding_query['query']['script_score']['script']['params']['text_eb']

        assert encoded_text == encoded_query

    end = time.time()

    non_cache_time = end - start

    loguru.logger.debug(
        f"Ccahe time: {cache_time}, Non-cache time: {non_cache_time}. Speed up %: {1 + (non_cache_time / (non_cache_time + cache_time))}")

    assert non_cache_time > cache_time


@pytest.mark.asyncio
async def test_embedding_queries(config_file_dict, nir_config_dict):
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

    i = 0

    for topic_num, topic in topics.items():

        query = query_obj.generate_query_embedding(topic_num=topic_num)

        q, res = await engine.execute_query(query, return_size=5,
                                            return_id_only=False)

        q_documents = [doc['_source'] for doc in res['hits']['hits']]
        assert len(q_documents) == 5
        assert q == query

        t_num, res = await engine.execute_query(query=None, topic_num=topic_num, return_size=5,
                                                return_id_only=False,
                                                query_type="embedding")

        t_documents = [doc['_source'] for doc in res['hits']['hits']]

        for doc1, doc2 in zip(q_documents, t_documents):
            assert doc1['Id'] == doc2['Id']
            assert doc1['Text'] == doc2['Text']

        assert topic_num == t_num

        i += 1
        if i > 100:
            break

    await client.close()


@pytest.mark.asyncio
async def test_pipeline_api(config_file_dict, nir_config_dict):
    p = NIRPipeline.build_from_config(config_fp=config_file_dict[0],
                                      engine="elasticsearch",
                                      nir_config_fp=nir_config_dict[0])

    results = await p.run_pipeline(cosine_offset=5.0)
