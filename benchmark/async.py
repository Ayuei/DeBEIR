import asyncio
import shutil
import tarfile

from debeir import Client, NIRPipeline


async def run_all_queries(client, p):
    for topic_num in p.engine.query.topics:
        body = p.engine.query.generate_query_embedding(topic_num, cosine_offset=100000)

        res = await client.search(
            index=p.engine.index_name, body=body, size=100
        )


if __name__ == "__main__":
    shutil.copy("./../tests/test_set.tar.gz", ".")
    with tarfile.open("./test_set.tar.gz", mode="r") as f:
        f.extractall()

    p = NIRPipeline.build_from_config(config_fp="./config.toml",
                                      engine="elasticsearch",
                                      nir_config_fp="./nir.toml")

    client = Client.build_from_config("elasticsearch", p.engine_config)

    asyncio.run(run_all_queries(client.es_client, p))
