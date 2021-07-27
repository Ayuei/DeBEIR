import dataclasses
from trectools import TrecRun, TrecEval, procedures, TrecQrel
#import modin.pandas as pd
import pandas as pd
from typing import Union, List
# Hijack for Recall @ K (which does not seem very useful)

#TrecRun.pd = pd
#TrecEval.pd = pd
#procedures.pd = pd
#TrecQrel.pd = pd

def get_recall(self, depth=1000, per_query=False, trec_eval=True, removeUnjudged=False):
    label = "Recall@%d" % (depth)

    run = self.run.run_data
    qrels = self.qrels.qrels_data

    # check number of queries
    nqueries = len(self.run.topics())

    if removeUnjudged:
        onlyjudged = pd.merge(run, qrels[["query", "docid", "rel"]], how="left")
        onlyjudged = onlyjudged[~onlyjudged["rel"].isnull()]
        run = onlyjudged[["query", "q0", "docid", "rank", "score", "system"]]

    if trec_eval:
        trecformat = self.run.run_data.sort_values(["query", "score", "docid"],
                                                   ascending=[True, False, False]).reset_index()
        topX = trecformat.groupby("query")[["query", "docid", "score"]].head(depth).reset_index(drop=True)
    else:
        topX = self.run.run_data.groupby("query")[["query", "docid", "score"]].head(depth).reset_index(drop=True)

    # gets the number of relevant documents per query
    n_relevant_docs = self.get_relevant_documents(per_query=True)

    relevant_docs = qrels[qrels.rel > 0]
    selection = pd.merge(topX, relevant_docs[["query", "docid", "rel"]], how="left")
    selection = selection[~selection["rel"].isnull()]

    recall_per_query = selection.groupby("query")["docid"].count() / n_relevant_docs
    recall_per_query.name = label
    recall_per_query = recall_per_query.reset_index().set_index("query")

    if per_query:
        return recall_per_query

    #if rprec_per_query.empty:
    #    return 0.0

    return (recall_per_query.sum() / nqueries)[label]


TrecEval.get_recall = get_recall


@dataclasses.dataclass(init=True, repr=True)
class Run:
    run: str
    fp: str
    metric: Union[float, int]


def parse_runs(fps: List, qrels=None, metric="NDCG", depth=10, kwargs=None):
    qrel = TrecQrel(qrels)
    parsed = []

    for arg in fps:
        parsed.append(parse_run(arg,
                                qrels=qrel,
                                metric=metric,
                                depth=depth))
    return parsed


def parse_run(fp, qrels, metric="NDCG", depth=10, kwargs=None):
    if kwargs is None:
        kwargs = {'per_query': True}

    run = TrecRun(fp)

    if not isinstance(qrels, TrecQrel):
        qrels = TrecQrel(qrels)

    evaluator = TrecEval(run, qrels)

    text_attrs = {
        "ndcg": 'get_ndcg',
        "p": 'get_precision',
        "r": 'get_relevant_documents',
        "recall": 'get_recall',
        "rprec": 'get_rprec',
        "map": 'get_map',
        'bpref': 'get_bpref'
    }

    if not metric.lower() == "r":
        kwargs['depth'] = depth

    res = getattr(evaluator, text_attrs[metric.lower()])(**kwargs)
    res = res.to_dict()

    res_key = list(res.keys())[0]
    res = res.pop(res_key)

    return Run(res, fp, metric + "@" + str(depth))
