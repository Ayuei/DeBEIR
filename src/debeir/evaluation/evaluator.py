from collections import defaultdict
from typing import Dict, List, Union

import loguru
from analysis_tools_ir import evaluate, sigtests
from debeir.core.config import GenericConfig, MetricsConfig


class Evaluator:
    """
    Evaluation class for computing metrics from TREC-style files
    """

    def __init__(self, qrels: str, metrics: List[str]):
        self.qrels = qrels
        self.metrics = []
        self.depths = []

        try:
            self._validate_and_setup_metrics(metrics)
        except AssertionError:
            raise ValueError("Metrics must be of the form metric@depth")

    def _validate_and_setup_metrics(self, metrics):
        for metric in metrics:
            assert "@" in metric
            try:
                metric, depth = metric.split("@")
            except:
                raise RuntimeError(f"Unable to parse metric {metric}")

            assert metric.isalpha()
            assert depth.isdigit()

            self.metrics.append(metric)
            self.depths.append(int(depth))

    def evaluate_runs(self, res: Union[str, List[str]], **kwargs):
        """
        Evaluates the TREC-style results from an input result list or file

        :param res: Results file path or raw results list
        :param kwargs: Keyword arguments to pass to the underlying analysis_tools_ir.parse_run library
        :return:
        """
        results = defaultdict(lambda: {})
        for metric, depth in zip(self.metrics, self.depths):
            results[metric][depth] = evaluate.parse_run(
                res, self.qrels,
                metric=metric, depth=depth,
                **kwargs
            )

        return results

    def average_all_metrics(self, runs: Dict, logger: loguru.logger):
        """
        Averages the metric per topic scores into a single averaged score.

        :param runs: Parsed run dictionary: {metric_name@depth: Run object}
        :param logger: Logger to print metrics
        """
        for metric, depth in zip(self.metrics, self.depths):
            run = runs[metric][depth].run
            logger.info(f"{metric}@{depth} Average: {sum(run.values()) / len(run):.4}")

    def sigtests(self, results_a, results_b):
        """
        Run a paired significance test on two result files

        :param results_a:
        :param results_b:
        :return:
        """
        return sigtests.paired.paired_t_test(results_a, results_b, self.qrels)

    @classmethod
    def build_from_config(cls, config: GenericConfig, metrics_config: MetricsConfig):
        return cls(config.qrels, metrics_config.metrics)
