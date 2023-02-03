"""
Results abstraction

The executor will return results in raw form, and needs to parsed to a common Results interface.

This result abstraction uses lazy evaluation and will only compute and manipulate documents from raw form when requested.
"""

from typing import List

from debeir.core.document import Document, document_factory

LAZY_STATIC_DOCUMENT_KEY = "document_objects"
LAZY_STATIC_DOCUMENT_TOPICS = "document_topics"
LAZY_STATIC_DOCUMENT_HASHMAP = "document_topics"


class Results:
    document_cls: Document

    def __init__(self, results: List, query_cls, engine_name):
        self.results = results
        self.document_cls: Document = document_factory[engine_name]
        self.__doc_cur = 0
        self.__topic_num = None
        self.lazy_static = {}
        self.query_cls = query_cls
        self.topic_flag = False

    def _as_documents(self, recompile=False):
        if recompile or 'document_objects' not in self.lazy_static:
            self.lazy_static[LAZY_STATIC_DOCUMENT_KEY] = self.document_cls.from_results(self.results,
                                                                                        self.query_cls,
                                                                                        ignore_facets=False)
            self.lazy_static[LAZY_STATIC_DOCUMENT_TOPICS] = list(self.lazy_static[LAZY_STATIC_DOCUMENT_KEY].keys())

        return self.lazy_static[LAZY_STATIC_DOCUMENT_KEY]

    def get_topic_ids(self):
        if LAZY_STATIC_DOCUMENT_KEY not in self.lazy_static:
            self._as_documents()

        return self.lazy_static[LAZY_STATIC_DOCUMENT_TOPICS]

    def __iter__(self):
        self._as_documents()
        self.__doc_cur = 0

        if not self.__topic_num:
            self.__topic_num = 0

        return self

    def __next__(self):
        if self.topic_flag:
            topic_num = self.__topic_num
        else:
            topic_num = self.get_topic_ids()[self.__topic_num]

        if self.__doc_cur >= len(self._as_documents()[topic_num]):
            self.__doc_cur = 0
            self.__topic_num += 1

            if self.topic_flag or self.__topic_num >= len(self.get_topic_ids()):
                raise StopIteration

            topic_num = self.get_topic_ids()[self.__topic_num]

        item = self._as_documents()[topic_num][self.__doc_cur]
        self.__doc_cur += 1

        return item

    def __call__(self, topic_num=None):
        self.__topic_num = topic_num
        if topic_num:
            self.topic_flag = True

        return self

    def __getitem__(self, item):
        return self._as_documents()[item]
