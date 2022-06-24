from typing import Dict

from classes.common.query import GenericElasticsearchQuery
from nir.parsers.parser import CSVParser


class BioRedditSubmissionParser(CSVParser):
    parse_fields = ["id", "body"]

    @classmethod
    def get_topics(cls, csvfile) -> Dict[int, Dict[str, str]]:
        return super().get_topics(csvfile)


class BioRedditCommentParser(CSVParser):
    parse_fields = ["id", "parent_id", "selftext", "title"]

    @classmethod
    def get_topics(cls, csvfile) -> Dict[str, Dict[str, str]]:
        topics = super().get_topics(csvfile)
        temp = {}

        for _, topic in topics.items():
            topic["text"] = topic.pop("selftext")
            topic["text2"] = topic.pop("title")
            temp[topic["id"]] = topic

        return temp


class BioRedditElasticsearchQuery(GenericElasticsearchQuery):
    def __init__(self, topics, config, *args, **kwargs):
        super().__init__(topics, config, *args, **kwargs)
        self.mappings = ["Text"]

        self.topics = topics
        self.config = config
        self.query_type = self.config.query_type

        self.embed_mappings = ["Text_Embedding"]

        self.query_funcs = {
            "query": self.generate_query,
            "embedding": self.generate_query_embedding,
        }
