from loguru import logger

from debeir import NIRPipeline

logger.disable("debeir")


def run_all_queries(p):
    for topic_num in p.engine.query.topics:
        p.engine.query.generate_query_embedding(topic_num)


if __name__ == "__main__":
    p = NIRPipeline.build_from_config(config_fp="./config.toml",
                                      engine="elasticsearch",
                                      nir_config_fp="./nir.toml")

    run_all_queries(p)
