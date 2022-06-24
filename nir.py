import argparse
import logging
import os
import sys
import asyncio
from typing import Type

import shutup;

from classes.common.executor import GenericExecutor
from classes.common.query import GenericElasticsearchQuery

shutup.please()

from elasticsearch import AsyncElasticsearch
from loguru import logger

from nir.classes.common.config import GenericConfig
from nir.classes.factory import apply_nir_config
from nir.evaluation.evaluator import Evaluator
from nir.classes.factory import factory_fn


@apply_nir_config
async def run_config_es(topics, config: GenericConfig,
                        config_fp: str,
                        executor_cls: Type[GenericExecutor],
                        query_obj: GenericElasticsearchQuery,
                        address="127.0.0.1",
                        es_port="9200",
                        query_type=None,
                        index_name=None,
                        norm_weight="2.15",
                        output_file=None,
                        remove=False,
                        size=1000,
                        evaluate=False,
                        **kwargs):

    if output_file is None:
        os.makedirs(name=f"{kwargs['output_directory']}/{config.index}", exist_ok=True)
        output_file = (
            f"{kwargs['output_directory']}/{config.index}/{config_fp.split('/')[-1].replace('.toml', '')}"
        )
        logger.info(f"Output file not specified, writing to: {output_file}")

    if os.path.exists(output_file) and not remove:
        logger.info(f"Output file exists: {output_file}. Exiting...")
        sys.exit(0)

    assert (
            query_type or config.query_type
    ), "At least config or argument must be provided for query type"

    index_name = config.index if config.index else index_name
    assert index_name is not None, "Must provide an index name somewhere"

    if norm_weight != "automatic":
        norm_weight = float(norm_weight)

    if remove:
        open(output_file, "w+").close()

    es = AsyncElasticsearch(f"http://{address}:{es_port}", request_timeout=1800)

    query_executor = executor_cls(
        topics=topics,
        client=es,
        index_name=index_name,
        output_file=output_file,
        return_size=size,
        query=query_obj,
        config=config,
    )

    if norm_weight == "automatic" or config.automatic is True:
        await query_executor.run_automatic_adjustment()

    logger.info(f"Running {config.query_type} queries")

    await query_executor.run_all_queries(
        serialize=True, query_type=query_type, norm_weight=norm_weight
    )

    await es.close()

    if evaluate:
        evaluator = Evaluator(
            config.qrels,
            metrics=kwargs['metrics'],
        )
        results = evaluator.evaluate_runs(output_file)
        evaluator.average_all_metrics(results, logger=logger)


def main(
        topics,
        configs,
        debug=False,
        **kwargs
):
    logger.remove()
    logger.add("logs/output.log", enqueue=True)

    if debug:
        logger.add(sys.stderr, level="DEBUG")
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logger.add(sys.stderr, level="INFO")
        shutup.mute_warnings()

    for config_fp in configs:
        query_cls, config, parsed_topics, executor_cls = factory_fn(topics, config_fp)
        if kwargs['elasticsearch']:
            asyncio.run(run_config_es(parsed_topics, config, config_fp, executor_cls, query_cls, **kwargs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the NIR model')
    parser.add_argument('--topics', help="Path to topic file")
    parser.add_argument('--configs', nargs='*',
                        help="Run Config to run, can add multiple configs")
    parser.add_argument('--nir_config', default="./configs/nir.toml",
                        help="NIR Server Config")
    parser.add_argument('--elasticsearch', action='store_true', help="Use the Elasticsearch Engine")
    parser.add_argument('--solr', action='store_true', help="Use the Lucene Solr Engine")
    parser.add_argument('--debug', action='store_true', help="Turn on debug logging")

    args = parser.parse_args()

    main(**vars(args))
