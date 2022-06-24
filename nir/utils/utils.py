def check_assertions_es(query_type, config):
    assert (
            query_type or config.query_type
    ), "At least config or argument must be provided for query type"


