import pytest
import os

from debeir.core import config
from debeir.datasets.factory import config_factory
from debeir.core.callbacks import EvaluationCallback, SerializationCallback
from debeir.core.config import _NIRMasterConfig
from debeir.evaluation.evaluator import Evaluator
from debeir.core.pipeline import NIRPipeline

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

    await p.run_pipeline(cosine_offset=5.0)

    assert cb.parsed_run is not None
    assert len(cb.parsed_run) > 1

    for metric in cb.parsed_run:
        assert cb.parsed_run[metric] is not None


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
        num_lines = 0
        cur_topic_num = None
        doc_itr = None
        for line in f:
            topic_num, _, doc_id, rank, score, _ = line.split()

            if cur_topic_num is None or topic_num != cur_topic_num:
                cur_topic_num = topic_num
                doc_itr = iter(results(topic_num))

            doc = next(doc_itr)

            assert str(doc.score) == score
            assert str(doc.doc_id) == doc_id
            assert str(doc.scores['rank']) == rank
            assert str(doc.topic_num) == topic_num

            num_lines += 1

    assert num_lines > 1000
