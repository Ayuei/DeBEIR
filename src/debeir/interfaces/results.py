from typing import Dict, List, Union
from debeir.interfaces.document import Document
from debeir.interfaces.document import document_factory


LAZY_STATIC_DOCUMENT_KEY = "document_objects"


class Results:
    document_cls: Document

    def __init__(self, results: List, engine_name):
        self.results = results
        self.document_cls: Document = document_factory[engine_name]
        self.__doc_cur = 0
        self.__topic_num = None
        self.__topic_nums = []
        self.lazy_static = {}

    def as_documents(self, recompile=False):
        if recompile or 'document_objects' not in self.lazy_static:
            self.lazy_static[LAZY_STATIC_DOCUMENT_KEY] = self.document_cls.from_results(self.results)

        return self.lazy_static[LAZY_STATIC_DOCUMENT_KEY]

    def __iter__(self):
        self.__doc_cur = 0

        if self.__topic_num is None:
            self.__topic_num = 0

        return self

    def __next__(self):
        if self.__topic_nums:
            topic_num = self.__topic_nums[self.__topic_num]
        else:
            topic_num = self.__topic_num

        if self.__doc_cur >= len(self.results[topic_num]):
            if self.__topic_nums and topic_num > len(self.__topic_nums):
                raise StopIteration

            self.__doc_cur = 0
            self.__topic_num += 1

        item = self.results[topic_num][self.__doc_cur]
        self.__doc_cur += 1

        return item

    def __call__(self, topic_num=None):
        self.__topic_num = topic_num

        return self

    def get_topic_nums(self):
        return self.results

    def __getitem__(self, item):
        return self.results[item]
