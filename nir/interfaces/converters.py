from collections import defaultdict
from typing import Dict, Union
from nir.interfaces.parser import Parser

import datasets

class ParsedTopicsToDataset:
    @classmethod
    def convert(cls, parser: Parser, output: Dict[Union[str, int], Dict]):
        """
        Convert from a Dict of shape
        {topic_id: {
                "Facet_1": ...
                "Facet_2": ...
            }
        }

        ->

        {
        topic_ids: [],
        Facet_1s: [],
        Facet_2s: [],
        }

        :param output: Topics output from the parser object
        :return:
        """
        flattened_topics = defaultdict(lambda: [])

        for topic_id, topic in output.items():
            flattened_topics["topic_id"].append(topic_id)

            for field in parser.parse_fields:
                if field in topic:
                    flattened_topics[field].append(topic[field])
                else:
                    flattened_topics[field].append(None)

        return datasets.Dataset.from_dict(flattened_topics)
