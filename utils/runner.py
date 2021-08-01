import dataclasses
from utils.query import Query, TestTrialsQuery
from utils.embeddings import Encoder
from typing import Dict, Optional, Union
from elasticsearch import AsyncElasticsearch as Elasticsearch
import asyncio
from asyncio import AbstractEventLoop
import tqdm.asyncio
import collections


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
                 test=False):
        self.topics = {"1": topics["1"]} if test else topics
        self.client = client
        self.index_name = index_name
        self.output_file = output_file
        self.return_size = return_size
        self.query = query
        self.encoder = encoder

    def generate_query(self, topic_num):
        raise NotImplementedError

    def execute_query(self, *args, **kwargs):
        raise NotImplementedError

    async def run_all_queries(self, serialize=False, query_type="query", *args, **kwargs):
        tasks = [self.execute_query(topic_num=topic_num, query_type=query_type, *args, **kwargs) for topic_num in self.topics]

        results = []

        for f in tqdm.asyncio.tqdm.as_completed(tasks):
            res = await f
            if serialize:
                self.serialise_results(*res)

    def serialise_results(self, topic_num, res, run_name="INSERT_RUN_NAME"):
        with open(self.output_file, "a+") as writer:
            for rank, result in enumerate(res["hits"]["hits"], start=1):
                doc = result["_source"]
                line = f"{topic_num}\tQ0\t{doc['IDInfo']['NctID']}\t{rank}\t{result['_score']}\t{run_name}\n"
                writer.write(line)


class ClinicalTrialsExecutor(Executor):
    query: TestTrialsQuery

    def __init__(self, topics: Dict[Union[str, int], Dict[str, str]], client: Elasticsearch, index_name: str,
                 output_file: str, query: Query, encoder: Optional[Encoder] = None, *args, **kwargs):

        super().__init__(topics, client, index_name, output_file, query, encoder)

        self.query_fns = {
            "query": self.generate_query,
            "ablation": self.generate_query_ablation,
            "embedding": self.generate_embedding_query
        }

    def generate_query(self, topic_num, best_fields=True, **kwargs):
        return self.query.generate_query(topic_num, best_fields=best_fields)

    def generate_query_ablation(self, topic_num, **kwargs):
        return self.query.generate_query_ablation(topic_num)

    def generate_embedding_query(self, topic_num, cosine_weights=None, query_weights=None,
                                 norm_weight=2.15, **kwargs):
        assert self.encoder is not None
        return self.query.generate_query_embedding(topic_num, self.encoder)

    async def execute_query(self, query=None, topic_num=None, ablation=False, query_type="query", **kwargs):
        if ablation:
            query_type = "ablation"

        if query:
            return await self.client.search(index=self.index_name, body=query, size=self.return_size)

        if topic_num:
            body = self.query_fns[query_type](topic_num=topic_num, **kwargs)
            res = await self.client.search(index=self.index_name, body=body, size=self.return_size)
            return [topic_num, res]
