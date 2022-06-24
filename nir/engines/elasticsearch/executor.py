import collections
import inspect
from typing import Dict, Optional, Union, List

import tqdm.asyncio
from elasticsearch import AsyncElasticsearch as Elasticsearch

from nir.utils.embeddings import Encoder
from classes.common.query import GenericElasticsearchQuery
from classes.common.config import apply_config


async def unpack_coroutine(f):
    res = await f
    while inspect.isawaitable(res):
        res = await res

    return res


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, None))
    return dict(items)


class ElasticsearchExecutor:
    def __init__(
        self,
        topics: Dict[Union[str, int], Dict[str, str]],
        client: Elasticsearch,
        index_name: str,
        output_file: str,
        query: GenericElasticsearchQuery,
        encoder: Optional[Encoder],
        return_size: int = 1000,
        test=False,
        return_id_only=True,
        config=None,
    ):
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
    async def run_all_queries(
        self, serialize=False, query_type="query", return_results=False, *args, **kwargs
    ) -> List:

        if not await self.client.ping():
            await self.client.close()
            raise RuntimeError(
                f"Elasticsearch instance cannot be reached at {self.client}"
            )

        tasks = [
            self.execute_query(
                topic_num=topic_num, query_type=query_type, *args, **kwargs
            )
            for topic_num in self.topics
        ]

        results = []

        for f in tqdm.asyncio.tqdm.as_completed(tasks, desc="Running Queries"):
            res = await unpack_coroutine(f)

            if serialize:
                self.serialise_results(*res)

            if return_results:
                results.append(res)

        return results

    def serialise_results(self, topic_num, res, run_name="INSERT_RUN_NAME"):
        with open(self.output_file, "a+") as writer:
            for rank, result in enumerate(res["hits"]["hits"], start=1):
                doc_id = None

                if self.return_id_only:
                    # doc_id = result["fields"]["IDInfo.NctID"][0]
                    doc_id = self.query.get_id_mapping(result["fields"])[0]
                else:
                    doc_id = self.query.get_id_mapping(result["_source"])

                line = f"{topic_num}\tQ0\t{doc_id}\t{rank}\t{result['_score']}\t{run_name}\n"
                writer.write(line)
