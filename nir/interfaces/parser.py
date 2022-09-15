import abc
import dataclasses
from collections import defaultdict
from dataclasses import dataclass
import csv
from typing import Dict, List
from xml.etree import ElementTree as ET
import dill
import json
import pandas as pd

# TODO: Parse fields can come from a config or ID_fields
# TODO: move _get_topics to private cls method with arguments, and expose get_topics as an instance method.


@dataclass(init=False)
class Parser:
    """
    Parser interface
    """

    parse_fields: List[str]

    @classmethod
    def normalize(cls, input_dict):
        return pd.io.json.json_normalize(input_dict,
                                         sep=".").to_dict(orient='records')[0]

    def get_topics(self, path, *args, **kwargs):
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

    # def _topic_iterator(self, all_topics):
    #    if self.topic_field_name:
    #        for topic in all_topics.findall(self.topic_field_name):
    #            yield topic

    #    elif self.parse_fields:
    #        for parse_field in self.parse_fields:
    #            yield all_topics.find(parse_field)
    @classmethod
    def _recurse_to_child_node(cls, node: ET.Element, track: List):
        if len(node.getchildren()) > 0:
            for child in node.getchildren():
                track.append(cls._recurse_to_child_node(child, track))

        return node

    @classmethod
    def unwrap(cls, doc_dict, key):
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


@dataclasses.dataclass(init=True)
class CSVParser(Parser):
    """
    Loads topics from a CSV file
    """
    parse_fields = ["id", "text"]

    @classmethod
    def _get_topics(cls, csvfile, *args, **kwargs) -> Dict[int, Dict[str, str]]:
        topics = {}
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            temp = {}

            for field in cls.parse_fields:
                temp[field] = row[field]

            topics[i] = temp

        return topics


@dataclasses.dataclass(init=True)
class JsonLinesParser(Parser):
    """
    Loads topics from a jsonl file,
    a JSON per line
    """
    parse_fields: List[str]
    id_field: str
    ignore_full_match: bool = True
    secondary_id: str = None

    def get_topics(self, path, *args, **kwargs) -> Dict[str, Dict[str, str]]:
        return self._get_topics(path, self.id_field, self.parse_fields, self.ignore_full_match,
                                self.secondary_id)

    @classmethod
    def _get_topics(cls, jsonlfile, id_field, parse_fields,
                    ignore_full_match=True, secondary_id=None, *args, **kwargs) -> Dict[str, Dict]:
        with open(jsonlfile, "r") as jsonl_f:
            topics = {}

            for jsonl in jsonl_f:
                json_dict = json.loads(jsonl)
                _id = json_dict.pop(id_field)

                if secondary_id:
                    _id = str(_id)+"_"+str(json_dict[secondary_id])

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
