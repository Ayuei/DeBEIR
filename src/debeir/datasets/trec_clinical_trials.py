import pathlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from debeir.core.parser import JsonLinesParser, XMLParser
from debeir.core.query import GenericElasticsearchQuery


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

TrecClinicalTrialsParser = XMLParser(
    parse_fields=None,
    id_field="number",
    topic_field_name="topic")


class TrecClincialElasticsearchQuery(GenericElasticsearchQuery):
    def __init__(self, topics, config, *args, **kwargs):
        super().__init__(topics, config, *args, **kwargs)

        # self.mappings = ['BriefTitle_Text',
        #                 'BriefSummary_Text',
        #                 'DetailedDescription_Text']

        self.mappings = [
            "BriefSummary_Text",
            "BriefTitle_Text",
            'DetailedDescription_Text',
            'Eligibility.Criteria.Textblock'
            'Eligibility.StudyPop.Textblock',
            'ConditionBrowse.MeshTerm',
            'InterventionBrowse.MeshTerm',
            'Condition',
            'Eligibility.Gender',
            "OfficialTitle"]

        self.topics = topics
        self.config = config
        self.query_type = self.config.query_type

        self.embed_mappings = ['BriefTitle_Embedding',
                               'BriefSummary_Embedding',
                               'DetailedDescription_Embedding']

        self.id_mapping = "docid"

        self.query_funcs = {
            "query": self.generate_query,
            "embedding": self.generate_query_embedding,
        }
