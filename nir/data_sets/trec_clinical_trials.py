import os
import pathlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from nir.interfaces.converters import ParsedTopicsToDataset
from nir.interfaces.parser import XMLParser, JsonLinesParser
from nir.training.utils import DatasetToSentTrans


class TREClinicalTrialDocumentParser(XMLParser):
    """
    Parser for Clinical Trials topics
    """

    parse_fields: List[str] = ["brief_title", "official_title",
                               "brief_summary", "detailed_description",
                               "eligibility", "condition_browse",
                               "intervention_browse"]
    topic_field_name: str
    id_field: str

    @classmethod
    def extract(cls, path) -> Dict:
        document = ET.parse(path).getroot()
        document_dict = defaultdict(lambda: defaultdict(lambda: []))
        document_dict['doc_id'] = pathlib.Path(path).parts[-1].strip(".xml")

        for parse_field in cls.parse_fields:
            node = document.find(parse_field)
            nodes: List[ET.Element] = []

            if node is not None:
                cls._recurse_to_child_node(node, nodes)

            if len(nodes) == 0 and node is not None:
                document_dict[parse_field] = node.text

            for node in nodes:
                text = node.text.strip()

                if not text:
                    continue

                if document_dict[parse_field][node.tag]:
                    document_dict[parse_field][node.tag].append(text)
                else:
                    document_dict[parse_field][node.tag] = [text]

            cls.unwrap(document_dict, parse_field)

        document_dict = pd.io.json.json_normalize(document_dict,
                                                  sep=".").to_dict(orient='records')[0]

        return document_dict


TrecClinicalTrialTripletParser = JsonLinesParser(
    parse_fields=["q_text", "brief_title", "official_title",
                  "brief_summary", "detailed_description", "rel"],
    id_field="qid",
    secondary_id="doc_id",
    ignore_full_match=True
)



