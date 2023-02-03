"""
Document abstraction.
Adaptor design pattern.

The pipeline must return results from the index as a Document object regardless of the engine type used.
"""

import abc
import dataclasses
from collections import defaultdict
from typing import Dict, List, Union

from debeir.utils.utils import flatten


@dataclasses.dataclass
class Document:
    """
    Generic Document class.
    Used as an interface for interacting across multiple indexes with different mappings.
    """
    doc_id: Union[int, float, str]
    topic_num: Union[int, str, float] = None
    facets: Dict = None
    score: Union[float, int] = 0.0  # Primay Score
    scores: Dict[str, Union[float, int]] = dataclasses.field(
        default_factory=lambda: {})  # Include other scores if needed

    @classmethod
    @abc.abstractmethod
    def from_results(cls, results, *args, **kwargs) -> Dict[Union[int, float], 'Document']:
        """
        Produces a list of Document objects from raw results returned from the index

        In the format {topic_num: [Document, ..., Document]}
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
            for k in keys:
                intermediate_repr = self._get_document_facet(intermediate_repr, k)

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

    def to_trec_format(self, rank, run_name) -> str:
        """
        Returns TREC format for the document
        :return:
            A trec formatted string
        """

        return f"{self.topic_num}\t" \
               f"Q0\t" \
               f"{self.doc_id}\t" \
               f"{rank}\t" \
               f"{self.score}\t" \
               f"{run_name}\n"

    @classmethod
    def get_trec_format(cls, ranked_list: List['Document'], run_name="NO_RUN_NAME", sort=True, sorting_func=None):
        """
        Get the trec format of a list of ranked documents. This function is a generator.

        :param ranked_list: A list of Document-type objects
        :param run_name: Run name to print in the TREC formatted string
        :param sort: Whether to sort the input list in descending order of score.
        :param sorting_func: Custom sorting function will be used if provided
        """

        if sort:
            if sorting_func:
                ranked_list = sorting_func(ranked_list)
            else:
                ranked_list.sort(key=lambda doc: doc.score, reverse=True)

        for rank, document in enumerate(ranked_list, start=1):
            yield document.to_trec_format(rank, run_name)


class ElasticsearchDocument(Document):
    """
    Elasticsearch class, this handles converting results to a document object
    """

    @classmethod
    def from_results(cls, results, query_cls, ignore_facets=True,
                     *args, **kwargs) -> Dict[Union[int, float], 'Document']:
        """
        Convert a set of elasticsearch results, to a dictionary of [Topic_num, list[Document]]

        :param results:
        :type results:
        :param query_cls:
        :type query_cls:
        :param ignore_facets:
        :type ignore_facets:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        documents = defaultdict(lambda: [])

        for (topic_num, res) in results:
            for rank, result in enumerate(res["hits"]["hits"], start=1):
                doc_id = query_cls.get_id_mapping(result["_source"])
                facets = {}

                if not ignore_facets:
                    facets = {k: v for (k, v) in result['_source'].items() if not k.startswith("_")}

                documents[topic_num].append(ElasticsearchDocument(doc_id,
                                                                  topic_num,
                                                                  facets=facets,
                                                                  score=float(result['_score'])))

                documents[topic_num][-1].scores['rank'] = rank

        return dict(documents)


document_factory = {
    "elasticsearch": ElasticsearchDocument
}
