from typing import Dict, Optional, Union, List

import tqdm.asyncio
from elasticsearch import AsyncElasticsearch as Elasticsearch

from debeir.rankers.transformer_sent_encoder import Encoder
from debeir.interfaces.query import GenericElasticsearchQuery
from debeir.interfaces.config import apply_config
from debeir.utils.utils import unpack_coroutine
from debeir.interfaces.document import document_factory
from debeir.interfaces.results import Results


class ElasticsearchExecutor:
    """
    Executes an elasticsearch query given the query generated from the config, topics and query class object.

    Computes regular patterns of queries expected from general IR topics and indexes.
    Includes:
        1. Reranking
        2. End-to-End Neural IR
        3. Statistical keyword matching
    """
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
        self.document_cls = document_factory['elasticsearch']

    def generate_query(self, topic_num):
        """
        Generates a query given a topic number from the list of topics

        :param topic_num:
        """
        raise NotImplementedError

    def execute_query(self, *args, **kwargs):
        """
        Execute a query given parameters

        :param args:
        :param kwargs:
        """
        raise NotImplementedError

    @apply_config
    def _update_kwargs(self, **kwargs):
        return kwargs

    async def run_all_queries(
        self, query_type=None, return_results=False,
            return_size: int = None, return_id_only: bool = False, **kwargs
    ) -> List:
        """
        A generic function that will asynchronously run all topics using the execute_query() method

        :param query_type: Which query to execute. Query_type determines which method is used to generate the queries
               from self.query.query_funcs: Dict[str, func]
        :param return_results: Whether to return raw results from the client. Useful for analysing results directly or
               for computing the BM25 scores for log normalization in NIR-style scoring
        :param return_size: Number of documents to return. Overrides the config value if exists.
        :param return_id_only: Return the ID of the document only, rather than the full source document.
        :param args: Arguments to pass to the execute_query method
        :param kwargs: Keyword arguments to pass to the execute_query method
        :return:
            A list of results if return_results = True else an empty list is returned.
        """
        if not await self.client.ping():
            await self.client.close()
            raise RuntimeError(
                f"Elasticsearch instance cannot be reached at {self.client}"
            )

        kwargs = self._update_kwargs(**kwargs)

        if return_size is None:
            return_size = self.return_size

        if return_id_only is None:
            return_id_only = self.return_id_only

        if query_type is None:
            query_type = self.config.query_type

        kwargs.pop('return_size', None)
        kwargs.pop('return_id_only', None)
        kwargs.pop('query_type', None)

        tasks = [
            self.execute_query(
                topic_num=topic_num,
                query_type=query_type,
                return_size=return_size,
                return_id_only=return_id_only,
                **kwargs
            )
            for topic_num in self.topics
        ]

        results = []

        for f in tqdm.asyncio.tqdm.as_completed(tasks, desc="Running Queries"):
            res = await unpack_coroutine(f)

            if return_results:
                results.append(res)

        return results
