from typing import Dict, Optional, Union, List

import tqdm.asyncio
from elasticsearch import AsyncElasticsearch as Elasticsearch

from rankers.transformer_sent_encoder import Encoder
from common.query import GenericElasticsearchQuery
from common.config import apply_config
from nir.utils.utils import unpack_coroutine


class ElasticsearchExecutor:
    """
    Executes an elasticsearch query given the query generated from the config, topics and query class object.

    Computes regular patterns of queries expected from general IR topics and indexes.
    Includes:
        1. Reranking
        2. End-to-End Neural IR
        3. Statistical keyword matching

    As well as:
        1. Serialisation of results to TREC-style format for further evaluation
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
    async def run_all_queries(
        self, serialize=False, query_type="query", return_results=False, *args, **kwargs
    ) -> List:
        """
        A generic function that will asynchronously run all topics using the execute_query() method

        :param serialize: Whether to serialize the results to TREC-style output.
               Output file must be passed to the constructor
        :param query_type: Which query to execute. Query_type determines which method is used to generate the queries
               from self.query.query_funcs: Dict[str, func]
        :param return_results: Whether to return raw results from the client. Useful for analysing results directly or
               for computing the BM25 scores for log normalization in NIR-style scoring
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

    def serialise_results(self, topic_num, res, run_name="NO_RUN_NAME"):
        """
        Serialize results to self.output_file in a TREC-style format
        :param topic_num: Topic number to serialize
        :param res: Raw elasticsearch result
        :param run_name: The run name for TREC-style runs (default: NO_RUN_NAME)
        """
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
