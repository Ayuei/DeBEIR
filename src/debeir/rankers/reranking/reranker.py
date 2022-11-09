"""
General re-ranking interfaces to be implemented by child classes.
"""

import abc
from typing import List, AnyStr, Union

from debeir.interfaces.document import Document


class ReRanker:
    """
    General interface for a reranking.

    Child classes should implement the abstract methods.

    """
    ranked_list: List

    def __init__(self, query, ranked_list: List, *args, **kwargs):
        self.ranked_list = ranked_list
        self.query = query

    @classmethod
    @abc.abstractmethod
    def _compute_scores(cls, document_repr):
        pass

    @classmethod
    @abc.abstractmethod
    def _get_document_representation(cls, document) -> (AnyStr, AnyStr):
        pass

    def rerank(self) -> List:
        """
        Re-rank the passed ranked list based on implemented private _compute_scores method.

        :param ranked_list:
        :return:
            A ranked list in descending order of the score field (which will be the last item in the list)
        """
        ranking = []

        for document in self.ranked_list:
            doc_id, doc_repr = self._get_document_representation(document)
            score = self._compute_scores(doc_repr)

            ranking.append([doc_id, doc_repr, score])

        ranking.sort(key=lambda k: k[-1], reverse=True)

        return ranking


class DocumentReRanker(ReRanker):
    """
    Reranking interface for a ranked list of Document objects.
    """

    def __init__(self, query, ranked_list: List[Document], *args, **kwargs):
        super().__init__(query, ranked_list, *args, **kwargs)

    @abc.abstractmethod
    def _compute_scores(self, document_repr):
        pass

    @classmethod
    def _get_document_representation(cls, document: Document) -> (Union[int, str, float], Document):
        return document.doc_id, document


class ReRankerPool:
    # Reranks per topic using threads.
    pass
