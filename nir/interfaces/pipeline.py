import abc
from typing import List

from nir.interfaces.callbacks import Callback
from nir.engines.client import Client
from nir.data_sets.factory import factory_fn, get_nir_config
from nir.evaluation.evaluator import Evaluator
from nir.interfaces.executor import GenericElasticsearchExecutor
from nir.interfaces.config import Config, NIRConfig
from loguru import logger

from nir.interfaces.config import GenericConfig


class Pipeline:
    pipeline_structure = ["parser", "query", "engine", "evaluator"]
    cannot_disable = ["parser", "query", "engine"]
    callbacks: List[Callback]

    def __init__(self, engine: GenericElasticsearchExecutor,
                 master_config: NIRConfig,
                 run_config: Config,
                 callbacks = None):

        self.engine = engine
        self.run_config = run_config
        self.master_config = master_config
        self.disable = {}

        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

    @classmethod
    def build_from_config(cls, nir_config_fp, engine, config_fp) -> 'Pipeline':
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
            engine=executor,
            master_config=nir_config,
            run_config=config
        )

    def disable(self, parts: list):
        for part in parts:
            if part in self.pipeline_structure and part not in self.cannot_disable:
                self.disable[part] = True
            else:
                logger.warning(f"Cannot disable {part} because it doesn't exist or is integral to the pipeline")

    @abc.abstractmethod
    async def run_pipeline(self, *args,
                           **kwargs):
        raise NotImplementedError()


class NIRPipeline(Pipeline):
    run_config: GenericConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def prehook(self):
        if self.run_config.automatic or self.run_config.norm_weight == "automatic":
            logger.info(f"Running initial BM25 for query adjustment")
            await self.engine.run_automatic_adjustment()

    async def run_engine(self, *args, **kwargs):
        # Run bm25 nir adjustment
        logger.info(f"Running {self.run_config.query_type} queries")

        return await self.engine.run_all_queries(*args, return_results=True, **kwargs)

    async def posthook(self, *args, **kwargs):
        pass

    async def run_pipeline(self, *args, return_results=False, **kwargs):
        for cb in self.callbacks:
            cb.before()

        await self.prehook()
        results = await self.run_engine(*args, **kwargs)

        for cb in self.callbacks:
            cb.after(results)

        if return_results:
            return results

    def register_callback(self, cb):
        self.callbacks.append(cb)
