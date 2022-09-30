import abc
import dataclasses
from typing import Union, Dict, List

from debeir.utils.utils import flatten


@dataclasses.dataclass
class Document:
    """
    Generic Document class.
    Used as an interface for interacting across multiple indexes with different mappings.
    """
    doc_id: Union[int, float, str]
    facets: Dict
    score: Union[float, int] = 0.0

    @classmethod
    @abc.abstractmethod
    def from_results(cls) -> List['Document']:
        """
        Produces a list of Document objects from raw results returned from the index
        """
        pass

    def get_document_id(self):
        """
        :return:
            self.doc_id
        """
        return self.doc_id

    def flatten_facets(self, *args, **kwargs):
        """
        Flattens multi-level internal document facets into a single level
            e.g. Doc['Upper']['Lower'] -> Doc['Upper_Lower']
        :param args:
        :param kwargs:
        """
        self.facets = flatten(self.facets, *args, **kwargs)

    @classmethod
    def _get_document_facet(cls, intermediate_repr, key):
        return intermediate_repr[key]

    def get_document_facet(self, key, sep="_"):
        """
        Retrieve a document facet
        Works for multidimensional keys or single
        :param key: Facet to retrieve
        :param sep: The seperator for multidimensional key
        :return:
            Returns the document facet given the key (field)
        """
        if sep in key:
            keys = key.split(sep)

            intermediate_repr = self.facets
            for key in keys:
                intermediate_repr = self._get_document_facet(intermediate_repr, key)

            return intermediate_repr

        return self.facets[key]

    def set(self, doc_id=None, facets=None, score=None, facet=None, facet_value=None) -> 'Document':
        """
        Set attributes of the object. Use keyword arguments to do so. Works as a builder class.
        doc.set(doc_id="123").set(facets={"title": "my title"})
        :param doc_id:
        :param facets:
        :param score:
        :param facet:
        :param facet_value:

        :return:
            Returns document object
        """
        if doc_id is not None:
            self.doc_id = doc_id

        if facets is not None:
            self.facets = facets

        if score is not None:
            self.score = score

        if facet is not None and facet_value is not None:
            self.facets[facet] = facet_value

        return self

    def _get_trec_format(self) -> str:
        """
        Returns TREC format for the document
        :return:
            A trec formatted string
        """
        return f"{self.score}"

    @classmethod
    def get_trec_format(cls, ranked_list: List['Document'], sort=True):
        """
        Get the trec format of a list of ranked documents. This function is a generator.

        :param ranked_list: A list of Document-type objects
        :param sort: Whether to sort the input list in descending order of score.
        """

        if sort:
            ranked_list.sort(key=lambda doc: doc.score, reverse=True)

        for document in ranked_list:
            yield document._get_trec_format()
