import abc
from typing import List, AnyStr


# Interface for a ranker
class ReRanker:
    ranked_list: List

    def __init__(self, ranked_list: List, *args, **kwargs):
        self.ranked_list = ranked_list

    @classmethod
    @abc.abstractmethod
    def _compute_scores(cls, document_repr):
        pass

    @classmethod
    @abc.abstractmethod
    def _get_document_representation(cls, document) -> (AnyStr, AnyStr):
        pass

    def rerank(self) -> List:
        return self.rrerank(self.ranked_list)

    @classmethod
    def rrerank(cls, ranked_list: List) -> List:
        ranking = []

        for document in ranked_list:
            doc_id, doc_repr = cls._get_document_representation(document)
            score = cls._compute_scores(doc_repr)

            ranking.append([doc_id, doc_repr, score])

        ranking.sort(key=lambda k: k[1])

        return ranking
