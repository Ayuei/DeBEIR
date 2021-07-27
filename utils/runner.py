import dataclasses
from typing import Dict, Optional
from elasticsearch import AsyncElasticsearch as Elasticsearch
import asyncio
from asyncio import AbstractEventLoop
from .query import Query, TestTrialsQuery
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
    def __init__(self, topics: Dict[int, Dict[str, str]], client: Elasticsearch, index_name: str,
                 output_file: str, query: Query, return_size: Optional[int]=1000, test=False):
        self.topics        = {"1": topics["1"]} if test else topics
        self.client        = client
        self.index_name    = index_name
        self.output_file   = output_file
        self.return_size   = return_size
        self.query         = query

    def generate_query(self, topic_num):
        raise NotImplementedError

    async def execute_query(self, query=None, topic_num=None, ablation=False):
        if query:
            return await self.client.search(index=self.index_name, body=query, size=self.return_size)
        if topic_num:
            body = self.query.generate_query_ablation(topic_num) if ablation else self.query.generate_query(topic_num)
            res = await self.client.search(index=self.index_name, body=body, size=self.return_size)
            return [topic_num, res]

    async def run_all_queries(self, serialize=False, ablation=False):
        tasks = [self.execute_query(topic_num=topic_num, ablation=ablation) for topic_num in self.topics]

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
    def generate_query(self, topic_num, best_fields=True):
        query = self.query.generate_query(topic_num, best_fields=True)
    def generate_query_ablation(self, topic_num):
        query = self.query.generate_query_ablation(topic_num)

