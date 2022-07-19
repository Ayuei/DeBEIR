from dataclasses import dataclass
import csv
from typing import Dict, List
from xml.etree import ElementTree as ET
import dill

# TODO: Parse fields can come from a config or ID_fields


@dataclass(init=False)
class Parser:
    """
    Parser interface
    """
    @classmethod
    def get_topics(cls, path) -> Dict[int, Dict[str, str]]:
        raise NotImplementedError


class PickleParser(Parser):
    """
    Load topics from a pickle file
    """
    @classmethod
    def get_topics(cls, path) -> Dict[int, Dict[str, str]]:
        return dill.load(path)


class XMLParser(Parser):
    """
    Load topics from an XML file
    """
    parse_fields: List[str]
    topic_field_name: str
    id_field: str

    @classmethod
    def get_topics(cls, path) -> Dict[int, Dict[str, str]]:
        all_topics = ET.parse(path).getroot()
        qtopics = {}

        for topic in all_topics.findall(cls.topic_field_name):
            temp = {}
            for field in cls.parse_fields:
                try:
                    temp[field] = topic.find(field).text
                except:
                    continue

            qtopics[int(topic.attrib[cls.id_field])] = temp

        return qtopics


class CSVParser(Parser):
    """
    Loads topics from a CSV file
    """
    parse_fields = ["id", "text"]

    @classmethod
    def get_topics(cls, csvfile) -> Dict[int, Dict[str, str]]:
        topics = {}
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            temp = {}

            for field in cls.parse_fields:
                temp[field] = row[field]

            topics[i] = temp

        return topics
