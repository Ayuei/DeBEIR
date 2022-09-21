from typing import Dict, Union, Optional

import loguru
from elasticsearch import AsyncElasticsearch as Elasticsearch

from nir.engines.client import Client
from nir.interfaces.query import GenericElasticsearchQuery, Query
from nir.engines.elasticsearch.executor import ElasticsearchExecutor
from nir.interfaces.config import apply_config, Config, NIRConfig, GenericConfig
from nir.rankers.transformer_sent_encoder import Encoder
from nir.utils.scaler import unpack_elasticsearch_scores


class GenericElasticsearchExecutor(ElasticsearchExecutor):
    """
    Generic Executor class for Elasticsearch
    """
    query: GenericElasticsearchQuery

    def __init__(
        self,
        topics: Dict[Union[str, int], Dict[str, str]],
        client: Elasticsearch,
        index_name: str,
        output_file: str,
        query: GenericElasticsearchQuery,
        encoder: Optional[Encoder] = None,
        config=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            topics,
            client,
            index_name,
            output_file,
            query,
            encoder,
            config=config,
            *args,
            **kwargs,
        )

        self.query_fns = {
            "query": self.generate_query,
            "embedding": self.generate_embedding_query,
        }

    def generate_query(self, topic_num, best_fields=True, **kwargs):
        """
        Generates a standard BM25 query given the topic number

        :param topic_num: Query topic number to generate
        :param best_fields: Whether to use a curated list of fields
        :param kwargs:
        :return:
        """
        return self.query.generate_query(topic_num, **kwargs)

    #def generate_query_ablation(self, topic_num, **kwargs):
    #    return self.query.generate_query_ablation(topic_num)

    def generate_embedding_query(
        self,
        topic_num,
        cosine_weights=None,
        query_weights=None,
        norm_weight=2.15,
        automatic_scores=None,
        **kwargs,
    ):
        """
        Executes an NIR-style query with combined scoring.

        :param topic_num:
        :param cosine_weights:
        :param query_weights:
        :param norm_weight:
        :param automatic_scores:
        :param kwargs:
        :return:
        """
        assert self.encoder is not None or self.config.encoder is not None

        if "encoder" not in kwargs:
            kwargs["encoder"] = self.encoder

        return self.query.generate_query_embedding(
            topic_num,
            cosine_weights=cosine_weights,
            query_weight=query_weights,
            norm_weight=norm_weight,
            automatic_scores=automatic_scores,
            **kwargs,
        )

    @apply_config
    async def execute_query(
        self, query, return_size: int, return_id_only: bool,
            topic_num=None, ablation=False, query_type="query",
            **kwargs
    ):
        """
        Executes a query using the underlying elasticsearch client.

        :param query:
        :param topic_num:
        :param ablation:
        :param query_type:
        :param return_size:
        :param return_id_only:
        :param kwargs:
        :return:
        """

        if ablation:
            query_type = "ablation"

        if query:
            if return_id_only:
                # query["fields"] = [self.query.id_mapping]
                # query["_source"] = False
                query["_source"] = [self.query.id_mapping]
            res = await self.client.search(
                index=self.index_name, body=query, size=return_size
            )

            return [query, res]

        if topic_num:
            loguru.logger.debug(query_type)
            body = self.query_fns[query_type](topic_num=topic_num, **kwargs)
            if self.return_id_only:
                loguru.logger.debug("Skip")
                body["_source"] = [self.query.id_mapping]

            loguru.logger.debug(body)
            res = await self.client.search(
                index=self.index_name, body=body, size=return_size
            )

            return [topic_num, res]

    async def run_automatic_adjustment(self):
        """
        Get the normalization constant to be used in NIR-style queries for all topics given an initial
        run of BM25 results.
        """
        loguru.logger.info("Running automatic BM25 weight adjustment")

        # Backup variables temporarily
        #size = self.return_size
        #self.return_size = 1
        #self.return_id_only = True
        #prev_qt = self.config.query_type
        #self.config.query_type = "query"

        results = await self.run_all_queries(query_type="query",
                                             return_results=True,
                                             return_size=1,
                                             return_id_only=True,
                                             serialize=False)

        results = unpack_elasticsearch_scores(results)
        self.query.set_bm25_scores(results)

    @classmethod
    def build_from_config(cls, topics: Dict, query_obj: GenericElasticsearchQuery, client,
                          config: GenericConfig, nir_config: NIRConfig):
        """
        Build an query executor engine from a config file.
        """

        return cls(
            topics=topics,
            client=client,
            config=config,
            index_name=config.index,
            output_file="",
            return_size=nir_config.return_size,
            query=query_obj
        )
