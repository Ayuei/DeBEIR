import time

from loguru import logger
from tqdm import tqdm

from debeir import Client, NIRPipeline

logger.disable("debeir")


def run_all_queries(client, p):
    tasks = []

    for topic_num in p.engine.query.topics:
        body = p.engine.query.generate_query_embedding(topic_num)
        tasks.append(
            {"index": p.engine.index_name, "body": body, "size": 10}
        )

    for task in tqdm(tasks):
        client.search(**task)


if __name__ == "__main__":
    tracker = []

    p = NIRPipeline.build_from_config(config_fp="./config.toml",
                                      engine="elasticsearch",
                                      nir_config_fp="./nir.toml")

    client = Client.build_from_config("elasticsearch_sync", p.engine_config)

    start = time.time()
    run_all_queries(client.es_client, p)
    end = time.time()

    print(end - start)
