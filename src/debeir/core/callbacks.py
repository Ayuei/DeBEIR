"""
Callbacks for before after running.
E.g. before is for setup
after is for evaluation/serialization etc
"""

import abc
import os
import tempfile
import uuid
from typing import List

import loguru

from debeir.core.config import GenericConfig, NIRConfig
from debeir.core.pipeline import Pipeline
from debeir.datasets.factory import query_factory
from debeir.evaluation.evaluator import Evaluator


class Callback:
    """Base Callback class, Do not use directly. Derive a child class."""

    def __init__(self):
        self.pipeline = None

    @abc.abstractmethod
    def before(self, pipeline: Pipeline):
        """
        This callback will be called before the pipeline begins querying the index.

        :param pipeline:
        :type pipeline:
        :return:
        :rtype:
        """
        pass

    @abc.abstractmethod
    def after(self, results: List):
        """
        This callback will be called after the pipeline has fetched all the results from the index.

        :param results:
        :type results:
        :return:
        :rtype:
        """
        pass


class SerializationCallback(Callback):
    """
    Serialization callback, writes results to a file after pipeline completes
    """

    def __init__(self, config: GenericConfig, nir_config: NIRConfig):
        super().__init__()
        self.config = config
        self.nir_config = nir_config
        self.output_file = None
        self.query_cls = query_factory[self.config.query_fn]

    def before(self, pipeline: Pipeline):
        """
        Check if output file exists

        :return:
            Output file path
        """

        self.pipeline = Pipeline

        output_file = self.config.output_file
        output_dir = os.path.join(self.nir_config.output_directory, self.config.index)

        if output_file is None:
            os.makedirs(name=output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, str(uuid.uuid4()))

            loguru.logger.info(f"Output file not specified, writing to: {output_file}")

        else:
            output_file = os.path.join(output_dir, output_file)

        if os.path.exists(output_file):
            if not self.config.overwrite_output_if_exists:
                raise RuntimeError("Directory exists and isn't explicitly overwritten "
                                   "in config with overwrite_output_if_exists=True")

            loguru.logger.info(f"Output file exists: {output_file}. Overwriting...")
            open(output_file, "w+").close()

        pipeline.output_file = output_file
        self.output_file = output_file

    def after(self, results: List):
        """
        Serialize results to self.output_file in a TREC-style format
        :param results:
        :type results:
        """

        self._after(results,
                    output_file=self.output_file,
                    run_name=self.config.run_name)

    @classmethod
    def _after(self, results: List, output_file, run_name=None):
        if run_name is None:
            run_name = "NO_RUN_NAME"

        with open(output_file, "a+t") as writer:
            for doc in results:
                line = f"{doc.topic_num}\t" \
                       f"Q0\t" \
                       f"{doc.doc_id}\t" \
                       f"{doc.scores['rank']}\t" \
                       f"{doc.score}\t" \
                       f"{run_name}\n"

                writer.write(line)


class EvaluationCallback(Callback):
    """
    Evaluation callback, runs after pipeline completes and evaluates using a TREC binary or python library
    """

    def __init__(self, evaluator: Evaluator, config):
        super().__init__()
        self.evaluator = evaluator
        self.config = config
        self.parsed_run = None

    def before(self, pipeline: Pipeline):
        self.pipeline = Pipeline

    def after(self, results: List, id_field="id"):
        if self.pipeline.output_file is None:
            directory_name = tempfile.mkdtemp()
            fn = str(uuid.uuid4())

            fp = os.path.join(directory_name, fn)

            query = query_factory[self.config.query_fn]
            query.id_field = id_field

            SerializationCallback._after(results,
                                         output_file=fp,
                                         run_name=self.config.run_name)

            self.pipeline.output_file = fp

        parsed_run = self.evaluator.evaluate_runs(self.pipeline.output_file,
                                                  disable_cache=True)
        self.parsed_run = parsed_run

        return self.parsed_run
