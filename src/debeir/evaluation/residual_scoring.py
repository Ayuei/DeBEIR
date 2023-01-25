import os
import subprocess
import tempfile
import uuid
from typing import Dict, List, Union

from debeir.evaluation.evaluator import Evaluator


# Remove all documents that exist in the training set
# Evaluate on remaining
# Normalize for result set length, cut off at ????


class ResidualEvaluator(Evaluator):
    """ Residual Scoring is the scoring of a subset of documents or the residiaul. The residual is created by removing documents from the collection and qrels.
    """

    def __init__(self, qrels: str, metrics: List[str], filter_ids: Dict[str, List[str]]):
        """ 
        Args:
            qrels (str): Path to qrels 
            metrics (List[str]): A list of metrics with depth e.g. NDCG@1000
            filter_ids (Dict[str, List[str]]): A list of IDs to remove from the collection given by Dict[Topic_num, [Docids]]
        """
        super().__init__(qrels, metrics)
        self.qrels_fp = qrels
        self.filter_ids = filter_ids

    def _filter_run(self, res: str):
        if self.filter_ids is None:
            return res

        tmpdir = tempfile.mkdtemp()
        tmpfp = os.path.join(tmpdir, str(uuid.uuid4()))

        writer = open(tmpfp, 'w+')

        with open(res) as out_file:
            for line in out_file:
                topic_num, _, doc_id, _, _, _ = line.split()
                if doc_id in self.filter_ids[topic_num]:
                    continue

                writer.write(line)

        writer.close()

        return tmpfp

    def evaluate_runs(self, res: Union[str, List[str]], with_trec_binary=False, **kwargs):
        """ Run the residual evaluation for the runs

        :param res: The results to run the evaluator against
        :param with_trec_binary: Use the TREC C binary instead of the default Python library, defaults to False
        :return: A dictionary of supplied metrics of the results against the qrels 
        """
        if with_trec_binary:
            return self._evaluate_with_binary(res, **kwargs)

        fp = self._filter_run(res)

        return super().evaluate_runs(fp, **kwargs)

    def _evaluate_with_binary(self, res, **kwargs):
        fp = self._filter_run(res)

        output = subprocess.check_output(["trec_eval", self.qrels_fp, fp]).decode()

        metrics = {}

        for line in str(output).split("\n"):
            try:
                metric, _, value = line.split()
                metrics[metric] = value
            except:
                continue

        return metrics
