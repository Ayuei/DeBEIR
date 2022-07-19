import string
from collections import defaultdict
from enum import Enum
from typing import List, Union


class InputExample:
    """
    Copied from Sentence Transformer Library
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = '', texts: List[str] = None, label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label

        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))

    def get_label(self):
        return self.label

    def __getitem__(self, key):
        if key == "label":
            return self.get_label()

        if key == "texts":
            return self.texts

        if key in ["guid", "id"]:
            return self.guid

        raise KeyError()

    @classmethod
    def to_dict(cls, data: List['InputExample']):
        text_len = len(data[0].texts)
        processed_data = defaultdict(lambda: [])

        for datum in data:
            # string.ascii_lowercase

            processed_data["id"].append(datum.guid)
            processed_data["label"].append(datum.get_label())

            for i in range(text_len):
                letter = string.ascii_lowercase[i]  # abcdefghi
                # processed_data[text_a] = ...
                processed_data[f"text_{letter}"].append(datum.texts[i])

        return processed_data

    @classmethod
    def from_parser_output(cls, data):
        pass


class RelevanceExample(InputExample):
    """
    Converts Relevance Labels to 0 - 1
    """

    def __init__(self, max_score=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_score = max_score

    def get_label(self):
        return self.relevance()

    def relevance(self):
        """
        :return:
            Returns a normalised score for relevance between 0 - 1
        """
        return self.label / self.max_score


class DatasetTypes(Enum):
    """
    A collection of common dataset types that is usable in the library.
    """
    List: "List"
    ListInputExample: "ListInputExample"
    ListDict: "ListDict"
    HuggingfaceDataset: "HuggingfaceDataset"