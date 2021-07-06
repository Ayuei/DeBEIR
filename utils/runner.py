import dataclasses
from typing import Dict
from elasticsearch import Client
from asyncio import loop


@dataclasses.dataclass(init=True)
class Executor:
    topics: Dict[int, Dict[str]]
    client: Client
    index_name: str
    loop: loop
    return_size: Optional[int]=1000

    def generate_query(self, topic_num):
        raise NotImplementedError

    async def execute_query(self, query=None, topic_num=None):
        if query:
            yield await self.client.search(index=self.index_name, body=query, size=size)
        if topic_num:
            yield await self.client.search(index=self.index_name, body=self.generate_query(topic_num), size=size)

    async def execute_all_queries(self):
        for topic_num in self.topics:
            yield await self.generate_query(topic_num=topic_num)


@dataclasses.dataclass(init=True)
class ClinicalTrialsExecutor(Executor):
    def generate_query(self, topic_num):
        should = {
            "should": []
        }
        for field in self.topics[topic_num]:
            should['should'].append({
                "match": self.topics[topic_num][field]
            })

        return {
            "query": {
                should
            }
        }
