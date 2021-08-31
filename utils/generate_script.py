import copy

base_script = {
    "lang": "painless",
    # Compute faster dot products as all vectors are unit length

    "source": None,
    "params": None,
}


def generate_source(fields):
    s = """
        if (_score < 1.0) {
            return 0.0;
        }

        def weights = params.weights;
        """

    variables = []

    for i, field in enumerate(fields):
        field = field.replace(".", '_') + '_Embedding'
        s += f"double {field}_score = doc['{field}'].size() == 0 ? 0 : weights[{i}]*cosineSimilarity(params.q_eb, '{field}') + weights[{i}] * params.offset;\n"

        variables.append(f"{field}_score")

    s = s.strip()

    s = s + "\n double embed_score = " + " + ".join(variables) + ";"
    s = s + " \n return params.disable_bm25 == true ? embed_score : embed_score + Math.log(_score)/Math.log(params.norm_weight);"

    return s


def check_params(params):
    assert 'q_eb' in params
    assert 'weights' in params
    assert 'offset' in params


def generate_script(fields=None, params=None, source_generator=generate_source):
    script = copy.deepcopy(base_script)
    check_params(params)

    script['lang'] = "painless"
    script['source'] = source_generator(fields)
    script['params'] = params

    return script
