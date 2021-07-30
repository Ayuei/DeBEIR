from dataclasses import dataclass
import csv
from typing import Dict


@dataclass(init=False)
class Parser:
    @classmethod
    def get_topics(cls, path) -> Dict[int, Dict[str, str]]:
        raise NotImplementedError


class CDS2021_Parser(Parser):
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

