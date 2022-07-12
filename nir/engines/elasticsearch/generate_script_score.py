import copy
from typing import Union, Dict
from nir.rankers.transformer_sent_encoder import EMBEDDING_DIM_SIZE

base_script = {
    "lang": "painless",
    # Compute faster dot products as all vectors are unit length
    "source": None,
    "params": None,
}


class SourceBuilder:
    """
    Builds Script Score source for NIR-style queries in elasticsearch
    Uses the painless language

    This is a string builder class
    """
    def __init__(self):
        self.s = ""
        self.i = 0
        self.variables = []

    def _add_line(self, line):
        self.s = self.s + line.strip() + "\n"

    def add_preamble(self):
        """
        Adds preamble to the internal string
        This will return the bm25 score if the normalization constant is below 0
        """
        self._add_line(
            """
            if (params.norm_weight < 0.0) {
                return _score;
            }
        """
        )

    def add_log_score(self, ignore_below_one=False) -> "SourceBuilder":
        """
        Adds the BM25 log score line
        :param ignore_below_one: Ignore all scores below 1.0 as Log(1) = 0. Otherwise, just ignore Log(0 and under).
        :return:
            SourceBuilder
        """
        if ignore_below_one:
            self._add_line(
                "def log_score = _score < 1.0 ? 0.0 : Math.log(_score)/Math.log(params.norm_weight);"
                # "def log_score = Math.log(_score)/Math.log(params.norm_weight);"
            )
        else:
            self._add_line(
                "def log_score = _score <= 0.0 ? 0.0 : Math.log(_score)/Math.log(params.norm_weight);"
                # "def log_score = Math.log(_score)/Math.log(params.norm_weight);"
            )

        return self

    def add_embed_field(self, qfield, field) -> "SourceBuilder":
        """
        Adds a cosine score line.
        :param qfield: Query field
        :param field: Document facet field
        :return:
        """
        if "embedding" not in field.lower():
            field = field.replace(".", "_") + "_Embedding"

        variable_name = f"{field}_{qfield}_score"

        self._add_line(
            f"double {variable_name} = doc['{field}'].size() < {EMBEDDING_DIM_SIZE} ? 0 : params.weights[{self.i}]*cosineSimilarity(params.{qfield}"
            f", '{field}') + params.offset; "
            # f"double {variable_name} = cosineSimilarity(params.{qfield}, '{field}') + 1.0; "
        )
        self.variables.append(variable_name)

        self.i += 1

        return self

    def finish(self):
        """
        Finalises the script score and returns the internal string
        :return:
            A string containing the script score query
        """
        self._add_line("double embed_score = " + " + ".join(self.variables) + ";")
        self._add_line(
            "return params.disable_bm25 == true ? log_score : embed_score + log_score;"
        )

        return self.s


def generate_source(qfields: Union[list, str], fields) -> str:
    """
    Generates the script source based off a set of input fields and facets

    :param qfields: Query fields (or topic fields)
    :param fields: Document facets to compute cosine similarity on
    :return:
    """
    sb = SourceBuilder()
    sb.add_log_score(ignore_below_one=True)

    if isinstance(qfields, str):
        qfields = [qfields]

    for qfield in qfields:
        for field in fields:
            sb.add_embed_field(qfield, field)

    s = sb.finish()

    return s


# def generate_source(fields, log_ignore=False):
#    s = ""
#
#    if log_ignore:
#
#    s = """
#        def log_score = _score < 1.0 ? _score : Math.log(_score)/Math.log(params.norm_weight);
#        def weights = params.weights;""".strip()+"\n"
#
#    variables = []
#
#    for i, field in enumerate(fields):
#        field = field.replace(".", '_') + '_Embedding'
#        s += f"double {field}_score = doc['{field}'].size() == 0 ? 0 : weights[{i}]*cosineSimilarity(params.q_eb, '{field}') + params.offset;\n"
#
#        variables.append(f"{field}_score")
#
#    s = s.strip()
#
#    s = s + "\n double embed_score = " + " + ".join(variables) + ";"
#    s = s + " \n return params.disable_bm25 == true ? embed_score : embed_score + Math.log(_score)/Math.log(params.norm_weight);"
#
#    return s


def check_params_is_valid(params, qfields):
    """
    Validate if the parameters for the script score passes a simple sanity check.

    :param params:
    :param qfields:
    """
    for qfield in qfields:
        assert qfield in params

    assert "weights" in params
    assert "offset" in params


def generate_script(
    fields, params, source_generator=generate_source, qfields="q_eb"
) -> Dict:
    """
    Parameters for creating the script

    :param fields: Document fields to search
    :param params: Parameters for the script
    :param source_generator:  Function that will generate the script
    :param qfields: Query fields to search from (topic facets)
    :return:
    """
    script = copy.deepcopy(base_script)
    check_params_is_valid(params, qfields)

    script["lang"] = "painless"
    script["source"] = source_generator(qfields, fields)
    script["params"] = params

    return script
