import argparse
import logging
import sys
import asyncio
from typing import Type

import shutup

from nir.interfaces.executor import GenericExecutor
from nir.interfaces.query import GenericElasticsearchQuery

from elasticsearch import AsyncElasticsearch
from loguru import logger

from nir.engines.client import Client
from nir.interfaces.config import GenericConfig
from nir.data_sets.factory import apply_nir_config
from nir.evaluation.evaluator import Evaluator
from nir.data_sets.factory import factory_fn
from nir.utils.utils import create_output_file


@apply_nir_config
async def run_config_es(topics, config: GenericConfig,
                        config_fp: str,
                        executor_cls: Type[GenericExecutor],
                        query_obj: GenericElasticsearchQuery,
                        client: Client,
                        query_type=None,
                        norm_weight="2.15",
                        output_file=None,
                        overwrite_output_if_exists=False,
                        size=1000,
                        evaluate=False,
                        **kwargs):
    """
    Use the NIR library on elasticsearch search engine given an NIR config, an Index config and a set of topics
    :param topics: Set of query topics to run
    :param config: Configuration file for the index
    :param config_fp: Configuration file path
    :param executor_cls: Which executor class is to be used
    :param query_obj: Query object to generate queries
    :param client: Generic client object to use for search
    :param query_type: Which query type to execute: embedding or bm25 only
    :param norm_weight: Normalization constant for NIR-style scoring
    :param output_file: Output file for results
    :param overwrite_output_if_exists: Overwrite the output file if it exists
    :param size: The number of documents to retrieve for each query topic from the index
    :param evaluate: Evaluate the results given the metric config
    :param kwargs: Miscellaneous parameters
    """
    if not client.es_client:
        # Initialise client object to be shared amongst all config runs
        client.es_client = AsyncElasticsearch(f"{kwargs['protocol']}://{kwargs['ip']}:{kwargs['port']}",
                                              timeout=kwargs['timeout'])

    output_file = create_output_file(config, config_fp, overwrite_output_if_exists, output_file, **kwargs)

    query_executor = executor_cls(
        topics=topics,
        client=client.es_client,
        index_name=config.index,
        output_file=output_file,
        return_size=size,
        query=query_obj,
        config=config,
    )

    if norm_weight == "automatic" or config.automatic is True:
        await query_executor.run_automatic_adjustment()

    logger.info(f"Running {config.query_type} queries")

    await query_executor.run_all_queries(
        serialize=True, query_type=query_type, norm_weight=float(norm_weight) if norm_weight.isdigit() else norm_weight,
    )

    if evaluate:
        evaluator = Evaluator(
            config.qrels,
            metrics=kwargs['metrics'],
        )
        parsed_run = evaluator.evaluate_runs(output_file, disable_cache=True)
        evaluator.average_all_metrics(parsed_run, logger=logger)


async def main(
        topics,
        configs,
        debug=False,
        **kwargs
):
    """
    Main function loop. Executes the passed config files and executes them all asynchronously.

    :param topics: Input topics (file path)
    :param configs: File paths to configs
    :param debug: Debug flag
    :param kwargs: Additional arguments from cmd args to pass to run configs
    """
    logger.remove()
    logger.add("logs/output.log", enqueue=True)

    client = Client()

    if debug:
        logger.add(sys.stderr, level="DEBUG")
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logger.add(sys.stderr, level="INFO")
        shutup.mute_warnings()

    for config_fp in configs:
        query_cls, config, parsed_topics, executor_cls = factory_fn(topics, config_fp)
        if kwargs['elasticsearch']:
            await run_config_es(parsed_topics, config, config_fp, executor_cls, query_cls, client=client,
                                engine='elasticsearch', **kwargs)

    await client.close()


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

    asyncio.run(main(**vars(args)))
