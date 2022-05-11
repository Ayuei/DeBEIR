import os
import sys

import plac
from elasticsearch import AsyncElasticsearch

from executor.evaluator import Evaluator
from query_builder.embeddings import Encoder
from utils.scaler import unpack_scores
from config.factory import factory_fn
import asyncio
from loguru import logger


@logger.catch
@plac.opt("topics_path", "Path to query topics", type=str)
@plac.opt("address", "Elasticsearch Address", type=str)
@plac.opt("es_port", "Elasticsearch Port Number", type=int)
@plac.opt("model_path", "Path to Sentence Encoder model", type=str)
@plac.opt("query_type", "Query type", type=str)
@plac.opt("norm_weight", "Norm Weight", type=str)
@plac.opt("index_name", "Name of Elasticsearch Index", type=str)
@plac.opt("output_file", "Output file name and/or path", type=str)
@plac.opt("config_path", "Path to Run Config File", type=str)
@plac.flg("delete", "Overwrite output file it exists")
@plac.opt("size", "Retrieved Input Size", type=int)
def main(
    topics_path,
    address=None,
    es_port=None,
    model_path=None,
    query_type=None,
    index_name=None,
    norm_weight="2.15",
    output_file=None,
    config_path=None,
    delete=False,
    size=1000,
):
    encoder = None
    results = None

    es = AsyncElasticsearch([{"host": address, "port": es_port}], timeout=1800)
    loop = asyncio.get_event_loop()

    if output_file is None:
        os.makedirs(name=f"outputs/{index_name}", exist_ok=True)
        output_file = (
            f"outputs/{index_name}/{config_path.split('/')[-1].replace('.toml', '')}"
        )
        logger.info(f"Output file not specified, writing to: {output_file}")

    if os.path.exists(output_file) and not delete:
        logger.info(f"Output file exists: {output_file}. Exiting...")
        sys.exit(0)

    query, config, parser, executor = factory_fn(topics, config_path, index_name)
    assert (
        query_type or config.query_type
    ), "At least config or argument must be provided for query type"

    index_name = config.index if config.index else index_name
    assert index_name is not None, "Must provide an index name somewhere"

    topics = parser.get_topics(open(topics_path))

    if model_path:
        encoder = Encoder(model_path)

    if norm_weight != "automatic":
        norm_weight = float(norm_weight)

    if delete:
        open(output_file, "w+").close()

    ex = executor(
        topics=topics,
        client=es,
        index_name=index_name,
        output_file=output_file,
        return_size=size,
        query=query,
        encoder=encoder,
        config=config,
    )

    if norm_weight == "automatic" or config.automatic is True:
        logger.info("Running trial queries to get automatic weight adjustment")
        ex.return_size = 1
        ex.return_id_only = True

        prev_qt = config.query_type
        config.query_type = "query"

        results = loop.run_until_complete(
            ex.run_all_queries(serialize=False, return_results=True)
        )

        results = unpack_scores(results)
        ex.return_size = size
        config.query_type = prev_qt

    loop.run_until_complete(
        ex.run_all_queries(
            serialize=True,
            query_type=query_type,
            norm_weight=norm_weight,
            automatic_scores=results,
        )
    )

    loop.run_until_complete(es.close())
    loop.close()

    if config.evaluate:
        evaluator = Evaluator(config.qrels)
        results = evaluator.evaluate_runs(output_file)


if __name__ == "__main__":
    logger.add("output.log", enqueue=True)
    plac.call(main)
