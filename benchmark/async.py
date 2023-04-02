import asyncio
import statistics
import time

from loguru import logger
from tqdm import tqdm

from debeir import Client, NIRPipeline

logger.disable("debeir")


async def run_all_queries(client, p):
    tasks = []

    for topic_num in p.engine.query.topics:
        body = p.engine.query.generate_query_embedding(topic_num)
        tasks.append(asyncio.create_task(
            client.search(
                index=p.engine.index_name, body=body, size=10
            )
        ))

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await fut


def print_stats(times):
    print(f"max: {max(times)}, min: {min(times)}, avg: {statistics.mean(times)}, "
          f"std: {statistics.stdev(times)}")


if __name__ == "__main__":
    tracker = []

    p = NIRPipeline.build_from_config(config_fp="./config.toml",
                                      engine="elasticsearch",
                                      nir_config_fp="./nir.toml")

    client = Client.build_from_config("elasticsearch", p.engine_config)

    start = time.time()
    asyncio.run(run_all_queries(client.es_client, p))
    end = time.time()

    print(end - start)
