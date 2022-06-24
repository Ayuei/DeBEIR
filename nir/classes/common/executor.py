from typing import Dict, Union, Optional

import loguru
from elasticsearch import AsyncElasticsearch as Elasticsearch

from nir.classes.common.query import GenericElasticsearchQuery
from nir.engines.elasticsearch.executor import ElasticsearchExecutor
from nir.classes.common.config import apply_config
from nir.utils.embeddings import Encoder
from utils.scaler import unpack_scores


class GenericExecutor(ElasticsearchExecutor):
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
        self, query=None, topic_num=None, ablation=False, query_type="query", **kwargs
    ):
        if ablation:
            query_type = "ablation"

        if query:
            if self.return_id_only:
                # query["fields"] = [self.query.id_mapping]
                # query["_source"] = False
                query["_source"] = [self.query.id_mapping]
            res = await self.client.search(
                index=self.index_name, body=query, size=self.return_size
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
                index=self.index_name, body=body, size=self.return_size
            )

            return [topic_num, res]

    async def run_automatic_adjustment(self):
        loguru.logger.info("Running automatic BM25 weight adjustment")

        # Backup variables temporarily
        size = self.return_size
        self.return_size = 1
        self.return_id_only = True
        prev_qt = self.config.query_type

        self.config.query_type = "query"

        results = await self.run_all_queries(serialize=False, return_results=True)

        results = unpack_scores(results)
        self.return_size = size
        self.config.query_type = prev_qt
        self.query.set_bm25_scores(results)
        self.return_id_only = False
