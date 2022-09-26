import pytest
import os

from debeir.interfaces import config
from debeir.data_sets.factory import config_factory
from debeir.interfaces.callbacks import EvaluationCallback, SerializationCallback
from debeir.interfaces.config import _NIRMasterConfig
from debeir.evaluation.evaluator import Evaluator
from debeir.interfaces.pipeline import NIRPipeline

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

    p.engine.query.id_mapping = "Id"

    p.register_callback(cb)

    results = await p.run_pipeline(cosine_offset=5.0)


@pytest.mark.asyncio
async def test_serialization_cb(config_file_dict, nir_config_dict):
    c = config.GenericConfig.from_toml(config_file_dict[0])
    master_config = config_factory(nir_config_dict[0], _NIRMasterConfig)

    cb = SerializationCallback(c, master_config.get_nir_settings(return_as_instance=True))

    p = NIRPipeline.build_from_config(config_fp=config_file_dict[0],
                                      engine="elasticsearch",
                                      nir_config_fp=nir_config_dict[0])

    p.engine.query.id_mapping = "Id"
    p.register_callback(cb)

    results = await p.run_pipeline(cosine_offset=5.0)

    assert cb.output_file == p.output_file
    assert os.path.exists(p.output_file)

    with open(p.output_file, "r") as f:
        assert len(f.readlines()) > 1
