"""
NIR Reranker

[Insert paper link here]
"""

import math
from typing import Dict, List

from debeir.core.document import Document
from debeir.rankers.reranking.reranker import DocumentReRanker
from debeir.rankers.transformer_sent_encoder import Encoder
from debeir.utils import scaler
from scipy import spatial
from tqdm import tqdm


class NIReRanker(DocumentReRanker):
    """
    Re-ranker which uses the NIR scoring method
        score = log(bm25)/log(z) + cosine_sum
    """

    def __init__(self, query, ranked_list: List[Document], encoder: Encoder,
                 distance_fn=spatial.distance.cosine, facets_weights: Dict = None,
                 presort=False, fields_to_encode=None,
                 *args, **kwargs):

        if presort:
            ranked_list.sort(key=lambda k: k.score)

        super().__init__(query, ranked_list, *args, **kwargs)
        self.encoder = encoder
        self.top_score = self._get_top_score()
        self.top_cosine_score = -1

        self.query_vec = self.encoder(self.query)
        self.distance_fn = distance_fn
        self.fields_to_encode = fields_to_encode

        if facets_weights:
            self.facets_weights = facets_weights
        else:
            self.facets_weights = {}

        # Compute all the cosine scores
        self.pre_calc = {}
        self.pre_calc_finished = False
        self.log_norm = None

    def _get_top_score(self):
        return self.ranked_list[0].score

    def _compute_scores_helper(self):
        for document in tqdm(self.ranked_list, desc="Calculating cosine scores"):
            facet_scores = {}
            for facet in self.fields_to_encode if self.fields_to_encode else document.facets:
                if "embedding" in facet.lower():
                    continue

                document_facet = document.facets[facet]
                facet_weight = self.facets_weights[document_facet] if facet in self.facets_weights else 1.0

                # Early exit
                if facet_weight == 0:
                    continue

                document_vec = self.encoder(document_facet)
                facet_scores[facet] = self.distance_fn(self.query_vec, document_vec) * facet_weight

                sum_score = sum(facet_scores.values())
                facet_scores["cosine_sum"] = sum_score

                self.top_cosine_score = max(self.top_cosine_score, sum_score)
                self.pre_calc[document.doc_id] = facet_scores

        self.pre_calc_finished = True

    def _compute_scores(self, document):
        if not self.pre_calc_finished:
            self._compute_scores_helper()
            self.log_norm = scaler.get_z_value(self.top_cosine_score, self.top_score)

        return math.log(document.score, self.log_norm) + self.pre_calc[document.doc_id]["cosine_sum"]
