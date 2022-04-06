import collections
from typing import Dict, Optional, Union

import tqdm.asyncio
from elasticsearch import AsyncElasticsearch as Elasticsearch

from utils.embeddings import Encoder
from utils.query import Query, TrialsQuery, GenericQuery
from utils.config import apply_config


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, None))
    return dict(items)


class Executor:
    def __init__(self, topics: Dict[Union[str, int], Dict[str, str]], client: Elasticsearch, index_name: str,
                 output_file: str, query: Query, encoder: Optional[Encoder] = None, return_size: Optional[int] = 1000,
                 test=False, return_id_only=True, config=None):
        self.topics = {"1": topics["1"]} if test else topics
        self.client = client
        self.index_name = index_name
        self.output_file = output_file
        self.return_size = return_size
        self.query = query
        self.encoder = encoder
        self.return_id_only = return_id_only
        self.config = config

    def generate_query(self, topic_num):
        raise NotImplementedError

    def execute_query(self, *args, **kwargs):
        raise NotImplementedError

    @apply_config
    async def run_all_queries(self, serialize=False, query_type="query", return_results=False, *args, **kwargs):
        tasks = [self.execute_query(topic_num=topic_num, query_type=query_type, *args, **kwargs) for topic_num in
                 self.topics]

        results = []

        for f in tqdm.asyncio.tqdm.as_completed(tasks):
            res = await f
            if serialize:
                self.serialise_results(*res)

            if return_results:
                results.append(res)

        if return_results:
            return results

    def serialise_results(self, topic_num, res, run_name="INSERT_RUN_NAME"):
        with open(self.output_file, "a+") as writer:
            for rank, result in enumerate(res["hits"]["hits"], start=1):
                doc_id = None
                if self.return_id_only:
                    doc_id = result['fields']['IDInfo.NctID'][0]
                else:
                    doc_id = result['_source']['IDInfo']['NctID']

                line = f"{topic_num}\tQ0\t{doc_id}\t{rank}\t{result['_score']}\t{run_name}\n"
                writer.write(line)


class ClinicalTrialsExecutor(Executor):
    query: TrialsQuery

    def __init__(self, topics: Dict[Union[str, int], Dict[str, str]], client: Elasticsearch, index_name: str,
                 output_file: str, query: Query, encoder: Optional[Encoder] = None, config=None, *args, **kwargs):

        super().__init__(topics, client, index_name, output_file, query, encoder, config=config, *args, **kwargs)

        self.query_fns = {
            "query": self.generate_query,
            "ablation": self.generate_query_ablation,
            "embedding": self.generate_embedding_query
        }

    def generate_query(self, topic_num, best_fields=True, **kwargs):
        return self.query.generate_query(topic_num, **kwargs)

    def generate_query_ablation(self, topic_num, **kwargs):
        return self.query.generate_query_ablation(topic_num)

    def generate_embedding_query(self, topic_num, cosine_weights=None, query_weights=None,
                                 norm_weight=2.15, automatic_scores=None, **kwargs):
        assert self.encoder is not None or self.config.encoder is not None

        if "encoder" not in kwargs:
            kwargs['encoder'] = self.encoder

        return self.query.generate_query_embedding(topic_num, cosine_weights=cosine_weights,
                                                   query_weight=query_weights, norm_weight=norm_weight,
                                                   automatic_scores=automatic_scores, **kwargs)

    @apply_config
    async def execute_query(self, query=None, topic_num=None, ablation=False, query_type="query",
                            **kwargs):
        if ablation:
            query_type = "ablation"

        if query:
            if self.return_id_only:
                query["fields"] = ["IDInfo.NctID"]
                query["_source"] = False
            res = await self.client.search(index=self.index_name, body=query, size=self.return_size)

            return [query, res]

        if topic_num:
            body = self.query_fns[query_type](topic_num=topic_num, **kwargs)
            if self.return_id_only:
                body["_source"] = False
                body["fields"] = ["IDInfo.NctID"]

            res = await self.client.search(index=self.index_name, body=body, size=self.return_size)
            return [topic_num, res]


class MarcoExecutor(ClinicalTrialsExecutor):
    query: GenericQuery

    def __init__(self, topics: Dict[Union[str, int], Dict[str, str]], client: Elasticsearch, index_name: str,
                 output_file: str, query: Query, encoder: Optional[Encoder] = None, config=None, *args, **kwargs):
        super().__init__(topics, client, index_name, output_file,
                         query, encoder, config=config, *args, **kwargs)

        self.query_fns = {
            "query": self.generate_query,
            "embedding": self.generate_embedding_query
        }

    def generate_query(self, topic_num, best_fields=True, **kwargs):
        return self.query.generate_query(topic_num)

    def generate_embedding_query(self, topic_num, cosine_weights=None, query_weights=None,
                                 norm_weight=2.15, automatic_scores=None, **kwargs):
        return super().generate_embedding_query(topic_num, cosine_weights=None, query_weights=None,
                                                norm_weight=2.15, automatic_scores=None, **kwargs)

    async def execute_query(self, query=None, topic_num=None, ablation=False, query_type="query",
                            **kwargs):
        return super().execute_query(query, topic_num, ablation, query_type, **kwargs)
