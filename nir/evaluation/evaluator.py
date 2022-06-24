import loguru
from typing import Union, List, Dict
from collections import defaultdict

from analysis_tools_ir import evaluate, sigtests


class Evaluator:
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

    def evaluate_runs(self, res: Union[str, List[str]]):
        results = defaultdict(lambda: {})
        for metric, depth in zip(self.metrics, self.depths):
            results[metric][depth] = evaluate.parse_run(
                res, self.qrels, metric=metric, depth=depth
            )

        return results

    def average_all_metrics(self, runs: Dict, logger: loguru.logger):
        for metric, depth in zip(self.metrics, self.depths):
            run = runs[metric][depth].run
            logger.info(f"{metric}@{depth} Average: {sum(run.values()) / len(run):.4}")

    def sigtests(self, results_a, results_b):
        return sigtests.paired.paired_t_test(results_a, results_b, self.qrels)
