import copy

base_script = {
    "lang": "painless",
    # Compute faster dot products as all vectors are unit length

    "source": None,
    "params": None,
}


def generate_source(fields):
    s = """
        def weights = params.weights;
        double embed_score = 
    """

    for i, field in enumerate(fields):
        s += f" weights[{i}]*dotProduct(params.q_eb, {field}) + weights[{i}] * params.offset + "

    s = s.strip("+").strip()

    return s + " \n return embed_score + Math.log(_score)/Math.log(params.norm_weight)"


def check_params(params):
    assert 'q_eb' in params
    assert 'weights' in params
    assert 'offset' in params


def generate_script(fields=None, params=None, source_generator=generate_source):
    script = copy.deepcopy(base_script)
    check_params(params)

    script['source'] = source_generator(fields)
    script['params'] = params
