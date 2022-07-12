from dataclasses import dataclass
from typing import Dict, Union, Optional

from elasticsearch import AsyncElasticsearch as Elasticsearch

from nir.common.config import GenericConfig
from nir.common.executor import GenericExecutor
from nir.common.query import GenericElasticsearchQuery
from nir.rankers.transformer_sent_encoder import Encoder


class MarcoExecutor(GenericExecutor):
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
        return self.query.generate_query(topic_num)

    def generate_embedding_query(
        self,
        topic_num,
        cosine_weights=None,
        query_weights=None,
        norm_weight=2.15,
        automatic_scores=None,
        **kwargs,
    ):
        return super().generate_embedding_query(
            topic_num,
            cosine_weights=cosine_weights,
            query_weights=query_weights,
            norm_weight=2.15,
            automatic_scores=None,
            **kwargs,
        )

    async def execute_query(
        self, query=None, topic_num=None, ablation=False, query_type="query", **kwargs
    ):
        return super().execute_query(
            query, topic_num, ablation, query_type=query_type, **kwargs
        )


@dataclass(init=True, unsafe_hash=True)
class MarcoQueryConfig(GenericConfig):
    def validate(self):
        if self.query_type == "embedding":
            assert (
                self.encoder_fp and self.encoder
            ), "Must provide encoder path for embedding model"
            assert self.norm_weight is not None or self.automatic is not None, (
                "Norm weight be " "specified or be automatic"
            )

    @classmethod
    def from_toml(cls, fp: str, *args, **kwargs) -> "MarcoQueryConfig":
        return super().from_toml(fp, cls, *args, **kwargs)

    @classmethod
    def from_dict(cls, **kwargs) -> "MarcoQueryConfig":
        return super().from_dict(cls, **kwargs)
