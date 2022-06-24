import dataclasses

import loguru

from typing import Dict

from classes.common.config import apply_config, GenericConfig
from engines.elasticsearch.generate_script_score import generate_script
from utils.scaler import get_z_value


@dataclasses.dataclass(init=True)
class Query:
    topics: Dict[int, Dict[str, str]]
    config: GenericConfig


class GenericElasticsearchQuery(Query):
    id_mapping: str = "id"

    def __init__(self, topics, config, top_bm25_scores=None, *args, **kwargs):
        super().__init__(topics, config)
        self.mappings = ["Text"]

        self.topics = topics
        self.config = config
        self.query_type = self.config.query_type

        self.embed_mappings = ["Text_Embedding"]

        self.query_funcs = {
            "query": self.generate_query,
            "embedding": self.generate_query_embedding,
        }

        self.top_bm25_scores = top_bm25_scores

    def _generate_base_query(self, topic_num):
        qfield = list(self.topics[topic_num].keys())[0]
        query = self.topics[topic_num][qfield]
        should = {"should": []}

        for i, field in enumerate(self.mappings):
            should["should"].append(
                {
                    "match": {
                        f"{field}": {
                            "query": query,
                        }
                    }
                }
            )

        return qfield, query, should

    def generate_query(self, topic_num, *args, **kwargs):
        _, _, should = self._generate_base_query(topic_num)

        query = {
            "query": {
                "bool": should,
            }
        }

        return query

    def set_bm25_scores(self, scores):
        self.top_bm25_scores = scores

    def has_bm25_scores(self):
        return self.has_bm25_scores is not None

    @apply_config
    def generate_query_embedding(
        self, topic_num, encoder, norm_weight=2.15, ablations=False, *args, **kwargs
    ):
        qfields = list(self.topics[topic_num].keys())
        should = {"should": []}

        if self.has_bm25_scores():
            norm_weight = get_z_value(
                cosine_ceiling=len(self.embed_mappings) * len(qfields),
                bm25_ceiling=self.top_bm25_scores[topic_num],
            )

        params = {
            "weights": [1] * (len(self.embed_mappings) * len(self.mappings)),
            "offset": 1.0,
            "norm_weight": norm_weight,
            "disable_bm25": ablations,
        }

        embed_fields = []

        for qfield in qfields:
            for field in self.mappings:
                should["should"].append(
                    {
                        "match": {
                            f"{field}": {
                                "query": self.topics[topic_num][qfield],
                            }
                        }
                    }
                )

            params[f"{qfield}_eb"] = encoder.encode(
                encoder, topic=self.topics[topic_num][qfield]
            )
            embed_fields.append(f"{qfield}_eb")

        query = {
            "query": {
                "script_score": {
                    "query": {
                        "bool": should,
                    },
                    "script": generate_script(
                        self.embed_mappings, params, qfields=embed_fields
                    ),
                }
            }
        }

        loguru.logger.debug(query)
        return query

    def get_id_mapping(self, hit):
        return hit[self.id_mapping]
