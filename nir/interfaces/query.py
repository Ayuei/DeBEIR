import dataclasses

import loguru

from typing import Dict, Union, Optional

from nir.interfaces.config import apply_config, GenericConfig
from nir.engines.elasticsearch.generate_script_score import generate_script
from nir.utils.scaler import get_z_value


@dataclasses.dataclass(init=True)
class Query:
    """
    A query interface class
    :param topics: Topics that the query will be composed of
    :param config: Config object that contains the settings for querying
    """
    topics: Dict[int, Dict[str, str]]
    config: GenericConfig


class GenericElasticsearchQuery(Query):
    """
    A generic elasticsearch query. Contains methods for NIR-style (embedding) queries and normal BM25 queries.
    Requires topics, configs to be included
    """
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
        """
        Generates a simple BM25 query based off the query facets. Searches over all the document facets.
        :param topic_num:
        :param args:
        :param kwargs:
        :return:
        """
        _, _, should = self._generate_base_query(topic_num)

        query = {
            "query": {
                "bool": should,
            }
        }

        return query

    def set_bm25_scores(self, scores: Dict[Union[str, int], Union[int, float]]):
        """
        Sets BM25 scores that are used for NIR-style scoring. The top BM25 score for each topic is used
        for log normalization.

        Score = log(bm25)/log(z) + embed_score
        :param scores: Top BM25 Scores of the form {topic_num: top_bm25_score}
        """
        self.top_bm25_scores = scores

    def has_bm25_scores(self):
        """
        Checks if BM25 scores have been set
        :return:
        """
        return self.top_bm25_scores is not None

    @apply_config
    def generate_query_embedding(
        self, topic_num, encoder, norm_weight=2.15, ablations=False, cosine_ceiling=Optional[float], *args, **kwargs
    ):
        """
        Generates an embedding script score query for Elasticsearch as part of the NIR scoring function.

        :param topic_num: The topic number to search for
        :param encoder: The encoder that will be used for encoding the topics
        :param norm_weight: The BM25 log normalization constant
        :param ablations: Whether to execute ablation style queries (i.e. one query facet
                          or one document facet at a time)
        :param cosine_ceiling: Cosine ceiling used for automatic z-log normalization parameter calculation
        :param args:
        :param kwargs: Pass disable_cache to disable encoder caching
        :return:
            An elasticsearch script_score query
        """

        qfields = list(self.topics[topic_num].keys())
        should = {"should": []}

        if self.has_bm25_scores():
            cosine_ceiling = len(self.embed_mappings) * len(qfields) if cosine_ceiling is None else cosine_ceiling
            norm_weight = get_z_value(
                cosine_ceiling=cosine_ceiling,
                bm25_ceiling=self.top_bm25_scores[topic_num],
            )
            loguru.logger.debug(f"Automatic norm_weight: {norm_weight}")

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
        """
        Get the document ID

        :param hit: The raw document result
        :return:
            The document's ID
        """
        return hit[self.id_mapping]
