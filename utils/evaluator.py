import tqdm
import pandas as pd
from glob import glob
from trec_wrapper import parse_run
from trectools import TrecQrel
import os


class Evaluator:
    def evaluate_runs(self, metric='NDCG', depth=20):
        return parse_run()
