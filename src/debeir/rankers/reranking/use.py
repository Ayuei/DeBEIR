from debeir.rankers.reranking.nir import NIReRanker


class USEReRanker(NIReRanker):
    """
    Re-ranks based on the cosine_sum rather the complete NIR scoring
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_scores(self, document):
        if not self.pre_calc_finished:
            self._compute_scores_helper()

        return self.pre_calc[document.doc_id]["cosine_sum"]
