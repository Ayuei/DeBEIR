import asyncio
import os
from collections import defaultdict
from copy import deepcopy

import pytest
from debeir.rankers.reranking.nir import NIReRanker

from debeir.core import config
from debeir.datasets.factory import config_factory
from debeir.core.config import _NIRMasterConfig
from debeir.core.pipeline import BM25Pipeline

from debeir.rankers.reranking.use import USEReRanker
from test_pipeline import config_file_dict, nir_config_dict

@pytest.mark.asyncio
async def test_nir_rerank_instance_method(config_file_dict, nir_config_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])
    master_config = config_factory(nir_config_dict[0], _NIRMasterConfig)

    p = BM25Pipeline.build_from_config(config_fp=config_file_dict[0],
                                      engine="elasticsearch",
                                      nir_config_fp=nir_config_dict[0])

    res = await p.run_pipeline(cosine_offset=5.0)

    topic_id = res.get_topic_ids()[0]
    ranked_list = res[topic_id]

    original_ranked_list = deepcopy(ranked_list)

    ranker = NIReRanker("A random test query.", ranked_list, c.encoder,
                        fields_to_encode=['Text'])
    ranking = ranker.rerank()

    collisions = 0
    scores_distribution_original = defaultdict(lambda: 0.0)
    scores_distribution_reranked = defaultdict(lambda: 0.0)

    for rank_repr, first_rank_repr in zip(ranking, original_ranked_list):
        _, doc, score = rank_repr

        scores_distribution_reranked[score] += 1

        if doc.doc_id == first_rank_repr.doc_id:
            collisions += 1

        scores_distribution_original[doc.score] += 1
        assert score != first_rank_repr.score
        assert score >= 0.0

    assert collisions < int(len(ranking) * 0.9)
    print()

@pytest.mark.asyncio
async def test_n_rerank_instance_method(config_file_dict, nir_config_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])
    master_config = config_factory(nir_config_dict[0], _NIRMasterConfig)

    p = BM25Pipeline.build_from_config(config_fp=config_file_dict[0],
                                       engine="elasticsearch",
                                       nir_config_fp=nir_config_dict[0])

    res = await p.run_pipeline(cosine_offset=5.0)

    topic_id = res.get_topic_ids()[0]
    ranked_list = res[topic_id]

    original_ranked_list = deepcopy(ranked_list)

    ranker = USEReRanker("A random test query.", ranked_list, c.encoder,
                        fields_to_encode=['Text'])
    ranking = ranker.rerank()

    collisions = 0
    scores_distribution_original = defaultdict(lambda: 0.0)
    scores_distribution_reranked = defaultdict(lambda: 0.0)

    for rank_repr, first_rank_repr in zip(ranking, original_ranked_list):
        _, doc, score = rank_repr

        scores_distribution_reranked[score] += 1

        if doc.doc_id == first_rank_repr.doc_id:
            collisions += 1

        scores_distribution_original[doc.score] += 1
        assert score != first_rank_repr.score
        assert score >= 0.0

    # At least one document should move positions
    # As Neural Ranking tends to be orthogonal to BM25
    assert collisions < int(len(ranking) * 0.9)
