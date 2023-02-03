"""
Parser interface and implemented parser classes for common file types: XML, CSV, TSV, plaintext
"""

import abc
import csv
import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from xml.etree import ElementTree as ET

import dill
import pandas as pd


# TODO: Parse fields can come from a config or ID_fields
# TODO: move _get_topics to private cls method with arguments, and expose get_topics as an instance method.


@dataclass(init=True)
class Parser:
    """
    Parser interface
    """

    id_field: object
    parse_fields: List[str]

    @classmethod
    def normalize(cls, input_dict) -> Dict:
        """
        Flatten the dictionary, i.e. from Dict[int, Dict] -> Dict[str, str_or_int]

        :param input_dict:
        :return:
        """
        return pd.io.json.json_normalize(input_dict,
                                         sep=".").to_dict(orient='records')[0]

    def get_topics(self, path, *args, **kwargs):
        """
        Instance method for getting topics, forwards instance self parameters to the _get_topics class method.
        """

        self_kwargs = vars(self)
        kwargs.update(self_kwargs)

        return self._get_topics(path, *args, **kwargs)

    @classmethod
    @abc.abstractmethod
    def _get_topics(cls, path, *args, **kwargs) -> Dict[int, Dict[str, str]]:
        raise NotImplementedError


@dataclasses.dataclass(init=True)
class PickleParser(Parser):
    """
    Load topics from a pickle file
    """

    @classmethod
    def _get_topics(cls, path, *args, **kwargs) -> Dict[int, Dict[str, str]]:
        return dill.load(path)


@dataclasses.dataclass(init=True)
class XMLParser(Parser):
    """
    Load topics from an XML file
    """
    topic_field_name: str
    id_field: str
    parse_fields: List[str]

    @classmethod
    def _recurse_to_child_node(cls, node: ET.Element, track: List):
        """
        Helper method to get all children nodes for text extraction in a xml.

        :param node: Current node
        :param track: List to track nodes
        :return:
        """
        if len(node.getchildren()) > 0:
            for child in node.getchildren():
                track.append(cls._recurse_to_child_node(child, track))

        return node

    @classmethod
    def unwrap(cls, doc_dict, key):
        """
        Converts defaultdict to dict and list of size 1 to just the element

        :param doc_dict:
        :param key:
        """
        if isinstance(doc_dict[key], defaultdict):
            doc_dict[key] = dict(doc_dict[key])

            for e_key in doc_dict[key]:
                cls.unwrap(doc_dict[key], e_key)

        if isinstance(doc_dict[key], list):
            if len(doc_dict[key]) == 1:
                doc_dict[key] = doc_dict[key][0]

    def _get_topics(self, path, *args, **kwargs) -> Dict[int, Dict[str, str]]:
        all_topics = ET.parse(path).getroot()
        qtopics = {}

        for topic in all_topics.findall(self.topic_field_name):
            _id = topic.attrib[self.id_field]
            if _id.isnumeric():
                _id = int(_id)

            if self.parse_fields:
                temp = {}
                for field in self.parse_fields:
                    try:
                        temp[field] = topic.find(field).text.strip()
                    except:
                        continue

                qtopics[_id] = temp
            else:
                #  The topic contains the text
                qtopics[_id] = {"query": topic.text.strip()}

        return qtopics


@dataclasses.dataclass
class CSVParser(Parser):
    """
    Loads topics from a CSV file
    """
    id_field = "id"
    parse_fields = ["text"]

    def __init__(self, id_field=None, parse_fields=None):
        if parse_fields is None:
            parse_fields = ["id", "text"]

        if id_field is None:
            id_field = "id"

        super().__init__(id_field, parse_fields)

    @classmethod
    def _get_topics(cls, csvfile, dialect="excel",
                    id_field: str = None,
                    parse_fields: List[str] = None,
                    *args, **kwargs) -> Dict[int, Dict[str, str]]:
        topics = {}

        if isinstance(csvfile, str):
            csvfile = open(csvfile, 'rt')

        if id_field is None:
            id_field = cls.id_field

        if parse_fields is None:
            parse_fields = cls.parse_fields

        reader = csv.DictReader(csvfile, dialect=dialect)
        for row in reader:
            temp = {}

            for field in parse_fields:
                temp[field] = row[field]

            topics[row[id_field]] = temp

        return topics


@dataclasses.dataclass(init=True)
class TSVParser(CSVParser):

    @classmethod
    def _get_topics(cls, tsvfile, *args, **kwargs) -> Dict[int, Dict[str, str]]:
        return CSVParser._get_topics(tsvfile, *args, dialect='excel-tab', **kwargs)


@dataclasses.dataclass(init=True)
class JsonLinesParser(Parser):
    """
    Loads topics from a jsonl file,
    a JSON per line

    Provide parse_fields, id_field and whether to ignore full matches on json keys
    secondary_id appends to the primary id as jsonlines are flattened structure and may contain duplicate ids.
    """
    parse_fields: List[str]
    id_field: str
    ignore_full_match: bool = True
    secondary_id: str = None

    @classmethod
    def _get_topics(cls, jsonlfile, id_field, parse_fields,
                    ignore_full_match=True, secondary_id=None, *args, **kwargs) -> Dict[str, Dict]:
        with open(jsonlfile, "r") as jsonl_f:
            topics = {}

            for jsonl in jsonl_f:
                json_dict = json.loads(jsonl)
                _id = json_dict.pop(id_field)

                if secondary_id:
                    _id = str(_id) + "_" + str(json_dict[secondary_id])

                for key in list(json_dict.keys()):
                    found = False
                    for _key in parse_fields:
                        if ignore_full_match:
                            if key in _key or key == _key or _key in key:
                                found = True
                        else:
                            if _key == key:
                                found = True
                    if not found:
                        json_dict.pop(key)

                topics[_id] = json_dict

        return topics
