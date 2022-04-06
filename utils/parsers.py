from dataclasses import dataclass
import csv
from typing import Dict


@dataclass(init=False)
class Parser:
    @classmethod
    def get_topics(cls, path) -> Dict[int, Dict[str, str]]:
        raise NotImplementedError


class CSVParser:
    parse_fields = ['id', 'text']

    @classmethod
    def get_topics(cls, csvfile) -> Dict[int, Dict[str, str]]:
        topics = {}
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            temp = {}

            for field in cls.parse_fields:
                temp[field] = row['field']

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
    parse_fields = ['id', 'body']

    @classmethod
    def get_topics(cls, csvfile) -> Dict[int, Dict[str, str]]:
        return super().get_topics(csvfile)


class BioRedditCommentParser(CSVParser):
    parse_fields = ['id', 'parent_id', 'selftext', 'title']

    @classmethod
    def get_topics(cls, csvfile) -> Dict[str, Dict[str, str]]:
        topics = super().get_topics(csvfile)
        temp = {}

        for _, topic in topics.items():
            topic['text'] = topic.pop('selftext')
            topic['text2'] = topic.pop('title')
            temp[topic['id']] = topic

        return temp
