import abc

from nir.engines.client import Client
from nir.data_sets.factory import factory_fn, get_nir_config
from nir.evaluation.evaluator import Evaluator
from nir.interfaces.executor import GenericElasticsearchExecutor
from nir.interfaces.config import Config, NIRConfig
from nir.utils.utils import create_output_file
from loguru import logger

from nir.interfaces.config import GenericConfig


class Pipeline:
    pipeline_structure = ["parser", "query", "engine", "evaluator"]
    cannot_disable = ["parser", "query", "engine"]

    def __init__(self, engine: GenericElasticsearchExecutor,
                 master_config: NIRConfig,
                 run_config: Config,
                 evaluator: Evaluator = None):

        self.engine = engine
        self.evaluator = evaluator
        self.run_config = run_config
        self.master_config = master_config
        self.disable = {}

    @classmethod
    def build_from_config(cls, nir_config_fp, engine, config_fp) -> 'Pipeline':
        query_cls, config, parser, executor_cls, evaluator_cls = factory_fn(config_fp)

        nir_config, search_engine_config, metrics_config = get_nir_config(nir_config_fp,
                                                                          engine, ignore_errors=False)

        client = Client.build_from_config(engine, search_engine_config)

        topics = parser.get_topics(open(config.topics_path))
        query = query_cls(topics=topics, query_type=config.query_type, config=config)

        executor = executor_cls.build_from_config(
            topics,
            query,
            client.get_client(engine),
            config,
            nir_config
        )

        evaluator = evaluator_cls.build_from_config(topics, metrics_config)

        return cls(
            engine=executor,
            evaluator=evaluator,
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

    async def prehook(self):
        if self.run_config.automatic or self.run_config.norm_weight == "automatic":
            logger.info(f"Running initial BM25 for query adjustment")
            await self.engine.run_automatic_adjustment()

    async def run_engine(self, *args, **kwargs):
        # Run bm25 nir adjustment
        logger.info(f"Running {self.run_config.query_type} queries")

        await self.engine.run_all_queries(serialize=True, *args, **kwargs)

    def post_run(self, *args, **kwargs):
        self.evaluator.evaluate()

    async def run_pipeline(self, *args, **kwargs):
        await self.prehook()
        await self.run_engine(*args, **kwargs)
        self.post_run(*args, **kwargs)
