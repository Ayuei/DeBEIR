from collections import defaultdict
from typing import List, Dict, Union

from nir.evaluation.evaluator import Evaluator
from datasets import Dataset
from nir.rankers.transformer_sent_encoder import Encoder
#from sentence_transformers.util import dot_score, cos_sim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


distance_fns = {
    "dot_score": np.dot,
    "cos_sim": cosine_similarity
}


class SentenceEvaluator(Evaluator):
    def __init__(self, model: Encoder, dataset: Dataset, parsed_topics: Dict[Union[str, int], Dict],
                 text_cols: List[str], query_cols: List[str], id_col: str,
                 distance_fn: str,
                 qrels: str, metrics: List[str]):
        super().__init__(qrels, metrics)
        self.encoder = model
        self.dataset = dataset
        self.parsed_topics = parsed_topics
        self.distance_fn = distance_fns[distance_fn]
        self.query_cols = query_cols
        self.text_cols = text_cols

        self._get_topic_embeddings(query_cols)
        self.document_ebs = self._get_document_embedding_and_mapping(id_col, text_cols)

    def _get_topic_embeddings(self, query_cols):
        for topic_num, topic in self.parsed_topics.items():
            for query_col in query_cols:
                query = topic[query_col]
                query_eb = self.encoder(query)

                topic[query_col+"_eb"] = query_eb

    def _get_document_embedding_and_mapping(self, id_col, text_cols):
        document_ebs = defaultdict(lambda: defaultdict(lambda: []))

        for datum in self.dataset:
            for text_col in text_cols:
                embedding = self.encoder(datum[text_col])
                topic_num, doc_id = datum[id_col].split("_")
                document_ebs[topic_num][doc_id].append([text_col, embedding])

        return document_ebs

    def _get_score(self, a, b, aggregate="sum"):
        scores = []

        aggs = {
            "max": max,
            "min": min,
            "sum": sum,
            "avg": lambda k: sum(k)/len(k)
        }

        if not isinstance(a[0], list):
            a = [a]

        if not isinstance(b[0], list):
            b = [b]

        for _a in a:
            for _b in b:
                scores.append(float(self.distance_fn(_a, _b)))

        return aggs[aggregate](scores)

    def produce_ranked_lists(self):
        # Store the indexes to access
        # For each topic, sort.

        topics = defaultdict(lambda: [])  # [document_id, score]

        for topic_num, doc_topics in self.document_ebs.items():
            for doc_id, document_repr in doc_topics.items():
                doc_txt_cols, doc_embeddings = list(zip(*document_repr))

                query_ebs = [self.parsed_topics[text_col+"_eb"] for text_col in self.text_cols]
                topics[topic_num].append([doc_id, self._get_score(query_ebs, doc_embeddings)])

        for topic_num in topics:
            topics[topic_num].sort(key=lambda k: k[1], reverse=True)

        return topics
