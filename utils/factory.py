from utils.config import MarcoQueryConfig, TrialsQueryConfig
from utils.query import MarcoQuery, TrialsQuery


config_factory = {
    "clinical_trials": TrialsQueryConfig,
    "test_trials": TrialsQueryConfig,
    "med-marco": MarcoQueryConfig
}

query_factory = {
    "clinical_trials": TrialsQuery,
    "test_trials": TrialsQuery,
    "med-marco":  MarcoQuery,
}


def get_index_name(config_fp):
    with open(config_fp, "r") as reader:
        for line in reader:
            if line.startswith("index"):
                line = line.replace('"', '')
                return line.split("=")[-1].strip()
    return None


def query_config_factory(topics, config_fp, index=None):
    if index is None:
        index = get_index_name(config_fp)
        assert index is not None, "Index must be provided in the config file or as an an argument"

    config_fct = config_factory[index]
    query_fct = query_factory[index]
    config = config_fct.from_toml(config_fp)
    query_type = config.query_type

    query = query_fct(topics=topics, query_type=config.query_type, config=config)

    return query, config
