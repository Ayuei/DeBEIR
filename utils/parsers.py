from dataclasses import dataclass
import csv

@dataclass(init=False)
class Parser:
    def get_topics(cls, path) -> Dict[int, Dict[str]]:
        raise NotImplementedError

class CDS2021_Parser(Parser):
    def get_topics(cls, path) -> Dict[int, Dict[str]]:
        topics = {}
        with open(csv.Reader(path)) as reader:
            for row in reader:
                _id = row["topic_num"]
                text = row["text"]

                topics[_id] = {"text": text}

        return topics
