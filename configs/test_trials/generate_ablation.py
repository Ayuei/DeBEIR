import copy

import toml

baseline_template = {
    "query_type": "query",
    "query_field_usage": None,
    "index": "test_trials",
}

embedding_template = {
    "query_type": "embedding",
    "query_field_usage": None,
    "embed_field_usage": None,
    "automatic": True,
    "encoder_fp": "./model/",
    "index": "test_trials",
}

template_dict = {"query": baseline_template, "embedding": embedding_template}

query_types = ["query", "embedding"]
field_usages = [
    "best_recall_fields",
    "all",
    "best_map_fields",
    "best_embed_fields",
    "sensible",
    "sensible_embed",
    "sensible_embed_safe",
]

translator_lst = [
    "recall",
    "all",
    "map",
    "best_embed",
    "sensible",
    "sensible_embed",
    "embed_safe",
]

translator = dict()

for k, v in zip(field_usage, translator_lst):
    translator[k] = v


def generate_name(query_type, query_field_usage, embed_field_usage=None):
    s = ""

    if query_type == "query":
        s += "baseline_"
    else:
        s += "embedding_"

    s += translator[query_field_usage]

    if embed_field_usage:
        s += "_" + translator[embed_field_usage]


def generate_from_template(query_type, query_field_usage, embed_field_usage=None):
    template = copy.deepcopy(template_dict[query_type])

    template["query_field_usage"] = query_field_usage

    if embed_field_usage:
        template["embed_field_usage"] = embed_field_usage

    return template


if __name__ == "__main__":
    for query_type in query_types:
        for query_field in field_usages:
            for embed_field in field_usages:
                generate_name(
                    query_type=query_type,
                    query_field_usage=query_field,
                    embed_field_usage=embed_field,
                )
