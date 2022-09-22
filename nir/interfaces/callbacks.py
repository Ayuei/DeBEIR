"""
Callbacks for before after running.
E.g. before is for setup
after is for evaluation/serialization etc
"""

import abc
import os
import uuid
from typing import List

import loguru

from nir.data_sets.factory import query_factory
from nir.evaluation.evaluator import Evaluator
from nir.interfaces.config import GenericConfig, NIRConfig


class Callback:
    @abc.abstractmethod
    def before(self):
        pass

    @abc.abstractmethod
    def after(self, results: List):
        pass


class SerializationCallback(Callback):
    def __init__(self, config: GenericConfig, nir_config: NIRConfig):
        self.config = config
        self.nir_config = nir_config
        self.output_file = None
        self.query_cls = query_factory[self.config.query_fn]

    def before(self):
        """
        Check if output file exists

        :return:
            Output file path
        """

        output_file = self.config.output_file
        output_dir = os.path.join(self.nir_config.output_directory, self.config.index)

        if output_file is None:
            os.makedirs(name=output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, str(uuid.uuid4()))

            loguru.logger.info(f"Output file not specified, writing to: {output_file}")

        output_file = os.path.join(output_dir, output_file)

        if os.path.exists(output_file):
            if not self.config.overwrite_output_if_exists:
                raise RuntimeError("Directory exists and isn't explicitly overwritten "
                                   "in config with overwrite_output_if_exists=True")

            loguru.logger.info(f"Output file exists: {output_file}. Overwriting...")
            open(output_file, "w+").close()

        self.output_file = output_file

    def after(self, results: List):
        """
        Serialize results to self.output_file in a TREC-style format
        :param topic_num: Topic number to serialize
        :param res: Raw elasticsearch result
        :param run_name: The run name for TREC-style runs (default: NO_RUN_NAME)
        """

        with open(self.output_file, "a+t") as writer:
            for (topic_num, res) in results:
                for rank, result in enumerate(res["hits"]["hits"], start=1):
                    doc_id = None

                    #if self.return_id_only:
                    #    doc_id = self.query.get_id_mapping(result["fields"])[0]
                    #else:
                    doc_id = self.query_cls.get_id_mapping(result["_source"])

                    line = f"{topic_num}\t" \
                           f"Q0\t" \
                           f"{doc_id}\t" \
                           f"{rank}\t" \
                           f"{result['_score']}\t" \
                           f"{self.config.run_name}\n"

                    writer.write(line)


class EvaluationCallback(Callback):
    def __init__(self, evaluator: Evaluator, config):
        self.evaluator = evaluator
        self.config = config
        self.parsed_run = None

    def before(self):
        pass

    def after(self, results: List):
        parsed_run = self.evaluator.evaluate_runs(results, disable_cache=True)
        self.parsed_run = parsed_run

        return parsed_run
