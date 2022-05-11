from dataclasses import dataclass
import csv
from typing import Dict, List
from xml.etree import ElementTree as ET
import dill


@dataclass(init=False)
class Parser:
    @classmethod
    def get_topics(cls, path) -> Dict[int, Dict[str, str]]:
        raise NotImplementedError


class PickleParser(Parser):
    @classmethod
    def get_topics(cls, path) -> Dict[int, Dict[str, str]]:
        return dill.load(path)


class XMLParser(Parser):
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
                    temp[field] = topic.find(field)
                except:
                    continue

            qtopics[int(topic.attrib[cls.id_field]) - 1] = temp

        return qtopics


class CSVParser(Parser):
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


class CDS2021Parser(Parser):
    @classmethod
    def get_topics(cls, csvfile) -> Dict[int, Dict[str, str]]:
        topics = {}
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                continue

            _id = row[0]
            text = row[1]

            topics[_id] = {"text": text}

        return topics


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


class TrecCovidParser(XMLParser):
    parse_fields = ["query", "question", "narrative"]
    topic_field_name = "topic"
    id_field = "number"

    @classmethod
    def get_topics(cls, xmlfile) -> Dict[int, Dict[str, str]]:
        return super().get_topics(xmlfile)
