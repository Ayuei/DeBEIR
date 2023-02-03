"""
This is the Pipeline which can be built from a configuration file. There are currently two supported pipelines:
`BM25Pipeline` and `NIRPipeline`.

See `configs/trec2022` for examples of cf configuration files.

An example usage of the NIR
```
    from debeir.core.pipeline import NIRPipeline

    p = NIRPipeline.build_from_config(config_fp="configs/trec2022/baseline.toml",
                                      engine="elasticsearch",
                                      nir_config_fp="configs/nir.toml")

    results = await p.run_pipeline()
```
"""

import abc
from typing import List

from loguru import logger

import debeir
from debeir.core.config import Config, GenericConfig
from debeir.core.executor import GenericElasticsearchExecutor
from debeir.core.results import Results
from debeir.datasets.factory import factory_fn, get_nir_config
from debeir.engines.client import Client


class Pipeline:
    """
    Base class to inherit for implementing custom Pipelines
    """
    pipeline_structure = ["parser", "query", "engine", "evaluator"]
    cannot_disable = ["parser", "query", "engine"]
    callbacks: List['debeir.core.callbacks.Callback']
    output_file = None

    def __init__(self, engine: GenericElasticsearchExecutor,
                 engine_name: str,
                 metrics_config,
                 engine_config,
                 nir_config,
                 run_config: Config,
                 callbacks=None):

        self.engine = engine
        self.engine_name = engine_name
        self.run_config = run_config
        self.metrics_config = metrics_config
        self.engine_config = engine_config
        self.nir_config = nir_config
        self.output_file = None
        self.disabled = {}

        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

    @classmethod
    def build_from_config(cls, nir_config_fp, engine, config_fp) -> 'Pipeline':
        """Builds af a pipeline object from configuration files"""
        query_cls, config, parser, executor_cls = factory_fn(config_fp)

        nir_config, search_engine_config, metrics_config = get_nir_config(nir_config_fp,
                                                                          engine=engine,
                                                                          ignore_errors=False)

        client = Client.build_from_config(engine, search_engine_config)
        topics = parser._get_topics(config.topics_path)

        query = query_cls(topics=topics, query_type=config.query_type, config=config)

        executor = executor_cls.build_from_config(
            topics,
            query,
            client.get_client(engine),
            config,
            nir_config
        )

        return cls(
            executor,
            engine,
            metrics_config,
            search_engine_config,
            nir_config,
            config
        )

    def disable(self, parts: list):
        """Disable certain parts of the pipeline to reduce slowdowns from unneeded operations.
            See "Pipeline.pipeline_structure" for components.
        """
        for part in parts:
            if part in self.pipeline_structure and part not in self.cannot_disable:
                self.disabled[part] = True
            else:
                logger.warning(f"Cannot disable {part} because it doesn't exist or is integral to the pipeline")

    @abc.abstractmethod
    async def run_pipeline(self, *args,
                           **kwargs):
        raise NotImplementedError()

    def register_callback(self, cb: 'debeir.core.callbacks.Callback'):
        """
        Add a callback to the pipeline

        :param cb:
        :type cb:
        :return:
        :rtype:
        """
        self.callbacks.append(cb)


class NIRPipeline(Pipeline):
    """
    The NIR Pipeline, this pipeline runs queries against the index using a custom scoring function that logarithmically
    normalises the cosine scores and BM25 scores.

    See paper for more details: https://arxiv.org/abs/2007.02492
    """
    run_config: GenericConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def prehook(self):
        """
        Runs prehooks for the pipeline. For the NIR pipeline, this involves running a baseline BM25 query to collect
        parameters for logarithmic normalization.
        """
        if self.run_config.automatic or self.run_config.norm_weight == "automatic":
            logger.info(f"Running initial BM25 for query adjustment")
            await self.engine.run_automatic_adjustment()

    async def run_engine(self, *args, **kwargs):
        """
        Runs the executor class to query the index and parse the results

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        # Run bm25 nir adjustment
        logger.info(f"Running {self.run_config.query_type} queries")

        return await self.engine.run_all_queries(*args, return_results=True, **kwargs)

    async def posthook(self, *args, **kwargs):
        pass

    async def run_pipeline(self, *args, return_results=False, **kwargs):
        """
        Execute the entire pipeline for NIR-style scoring, including callbacks

        :param args:
        :type args:
        :param return_results: Whether to return the results from the pipeline, otherwise an empty Result set is returned
        :type return_results:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        for cb in self.callbacks:
            cb.before(self)

        await self.prehook()
        results = await self.run_engine(*args, **kwargs)
        results = Results(results, self.engine.query, self.engine_name)

        for cb in self.callbacks:
            cb.after(results)

        return results


class BM25Pipeline(NIRPipeline):
    """
    The BM25 Pipeline, this is an IR standard baseline. This is best used in conjunction with
    `debeir.engines.elasticsearch.change_bm25.change_bm25_params` for changing the k1 and b hyperparameters for the BM25
    scoring model.
    """

    async def run_pipeline(self, *args, return_results=False, **kwargs):
        """
        Run the BM25 pipeline, this does not use the embedding inverted index.

        :param args:
        :type args:
        :param return_results:
        :type return_results:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        for cb in self.callbacks:
            cb.before(self)

        results = await self.engine.run_all_queries(query_type="query",
                                                    return_results=True)

        results = Results(results, self.engine.query, self.engine_name)

        for cb in self.callbacks:
            cb.after(results)

        return results
