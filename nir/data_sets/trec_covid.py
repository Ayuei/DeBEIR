from typing import Dict

from nir.interfaces.query import GenericElasticsearchQuery
from nir.interfaces.parser import XMLParser


class TrecCovidParser(XMLParser):
    parse_fields = ["query", "question", "narrative"]
    topic_field_name = "topic"
    id_field = "number"

    @classmethod
    def get_topics(cls, xmlfile) -> Dict[int, Dict[str, str]]:
        return super().get_topics(xmlfile)


class TrecElasticsearchQuery(GenericElasticsearchQuery):
    def __init__(self, topics, config, *args, **kwargs):
        super().__init__(topics, config, *args, **kwargs)

        self.mappings = ["title", "abstract", "fulltext"]

        self.topics = topics
        self.config = config
        self.query_type = self.config.query_type

        self.embed_mappings = [
            "title_embedding",
            "abstract_embedding",
            "fulltext_embedding",
        ]

        self.id_mapping = "id"

        self.query_funcs = {
            "query": self.generate_query,
            "embedding": self.generate_query_embedding,
        }
