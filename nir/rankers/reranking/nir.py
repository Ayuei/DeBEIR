"""
NIR Reranker

[Insert paper link here]
"""

from typing import List, AnyStr, Dict

from tqdm import tqdm

from nir.utils import scaler
from nir.interfaces.document import Document
from nir.rankers.reranking.reranker import DocumentReRanker
from nir.rankers.transformer_sent_encoder import Encoder
from scipy import spatial
import math


class NIReRanker(DocumentReRanker):
    """
    Re-ranker which uses the NIR scoring method
        score = log(bm25)/log(z) + cosine_sum
    """

    def __init__(self, query, ranked_list: List[Document], encoder: Encoder,
                 distance_fn=spatial.distance.cosine, facets_weights: Dict = None,
                 *args, **kwargs):
        super().__init__(query, ranked_list, *args, **kwargs)
        self.encoder = encoder
        self.top_score = self._get_top_score()
        self.top_cosine_score = -1

        self.query_vec = self.encoder(self.query)
        self.distance_fn = distance_fn

        # Compute all the cosine scores
        self.pre_calc = {}
        self._compute_scores_helper()
        self.log_norm = scaler.get_z_value(self.top_score, self.top_cosine_score)
        self.facets_weights = facets_weights

    def _get_top_score(self):
        return self.ranked_list[0].score

    def _compute_scores_helper(self):
        for document in tqdm(self.ranked_list, desc="Calculating cosine scores"):
            facet_scores = {}
            for facet in document.facets:
                document_vec = self.encoder(facet)

                facet_weight = self.facets_weights[facet] if facet in self.facets_weights else 1.0
                facet_scores[facet] = self.distance_fn(self.query_vec, document_vec) * facet_weight

                sum_score = sum(facet_scores.values())
                facet_scores["cosine_sum"] = sum_score

                self.top_score = max(self.top_score, sum_score)
                self.pre_calc[document.doc_id] = facet_scores

    def _compute_scores(self, document):
        return math.log(document.score, self.log_norm) + self.pre_calc[document.id]["cosine_sum"]
