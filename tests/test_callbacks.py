import pytest
import pytest_asyncio

from interfaces import config
from nir.data_sets.factory import config_factory
from nir.interfaces.callbacks import EvaluationCallback
from nir.interfaces.config import _NIRMasterConfig
from nir.evaluation.evaluator import Evaluator
from nir.interfaces.pipeline import NIRPipeline
from test_pipeline import config_file_dict, nir_config_dict


@pytest.mark.asyncio
async def test_evaluation_cb(config_file_dict, nir_config_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])
    master_config = config_factory(nir_config_dict[0], _NIRMasterConfig)
    metrics_config = master_config.get_metrics(return_as_instance=True)

    evaluator = Evaluator.build_from_config(c, metrics_config=metrics_config)
    cb = EvaluationCallback(evaluator, config=c)

    p = NIRPipeline.build_from_config(config_fp=config_file_dict[0],
                                      engine="elasticsearch",
                                      nir_config_fp=nir_config_dict[0])

    p.register_callback(cb)

    results = await p.run_pipeline(cosine_offset=5.0)


def test_serialization_cb(config_file_dict):
    pass
