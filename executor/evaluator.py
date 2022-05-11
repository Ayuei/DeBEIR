import tqdm
import pandas as pd
from glob import glob
from trectools import TrecQrel
import os
from typing import Union, List, overload

from analysis_tools_ir import evaluate, sigtests


class Evaluator:
    def __init__(self, qrels: str, metrics: List[str], depths: List[int]):
        self.qrels = qrels
        self.metrics = metrics
        self.depths = depths

    @overload
    def evaluate_runs(self, results: Union[str, List[str]]):
        results = {}
        for metric, depth in zip(self.metrics, self.depths):
            results[metric][depth] = evaluate.parse_run(results, self.qrels, metric=metric, depth=depth)

    def evaluate_runs(self, results: Union[str, List[str]], metric="NDCG", depth=20):
        return evaluate.parse_run(results, self.qrels, metric=metric, depth=depth)

    def sigtests(self, results_a, results_b):
        return sigtests.paired.paired_t_test(results_a, results_b, self.qrels)

