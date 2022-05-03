import copy

base_script = {
    "lang": "painless",
    # Compute faster dot products as all vectors are unit length

    "source": None,
    "params": None,
}


class SourceBuilder:
    def __init__(self):
        self.s = ""
        self.i = 0
        self.variables = []

    def _add_line(self, line):
        self.s = self.s + line.strip() + "\n"

    def add_preamble(self):
        self._add_line("""
            if (params.norm_weight < 0.0) {
                return _score;
            }
        """)

    def add_log_score(self, ignore_below_one=False) -> 'SourceBuilder':
        if ignore_below_one:
            self._add_line("def log_score = _score < 1.0 ? 0.0 : Math.log(_score)/Math.log(params.norm_weight);")
        else:
            self._add_line("def log_score = _score <= 0.0 ? 0.0 : Math.log(_score)/Math.log(params.norm_weight);")

        return self

    def add_embed_field(self, field) -> 'SourceBuilder':
        if "Embedding" not in field:
            field = field.replace(".", '_') + '_Embedding'

        self._add_line(f"double {field}_score = doc['{field}'].size() == 0 ? 0 : weights[{self.i}]*cosineSimilarity(params.q_eb, '{field}') + params.offset;")
        self.variables.append(f"{field}_score")

        self.i += 1

        return self

    def finish(self):
        self._add_line("double embed_score = " + " + ".join(self.variables) + ";")
        self._add_line("return params.disable_bm25 == true ? log_score : embed_score + log_score;")

        return self.s


def generate_source(fields) -> str:

    sb = SourceBuilder()
    sb.add_log_score(ignore_below_one=True)
    for field in fields:
        sb.add_embed_field(field)

    s = sb.finish()

    return s

#def generate_source(fields, log_ignore=False):
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


def check_params(params):
    assert 'q_eb' in params
    assert 'weights' in params
    assert 'offset' in params


def generate_script(fields, params, source_generator=generate_source):
    script = copy.deepcopy(base_script)
    check_params(params)

    script['lang'] = "painless"
    script['source'] = source_generator(fields)
    script['params'] = params

    return script
