def unpack_scores(results):
    scores = {}
    if isinstance(results[0][0], list):
        results = results[0]

    for raw_result in results:
        topic_num, result = raw_result
        for res in result["hits"]["hits"]:
            score = float(res['_score'])

            scores[topic_num] = score

    return scores


def get_z_value(cosine_ceiling, bm25_ceiling):
    return bm25_ceiling ** (1 / float(cosine_ceiling))


#class Scaler:
#    def __init__(self, gold_standard, qwt, cwt):
#        self.scores = []
#
#    def get_norm_weight_by_query(self, qid, estimate_ceiling=False):
#        return self.get_norm_weight(self.qwt, self.cwt, bm25_ceiling=self.scores[int(qid) - 1],
#                                    estimate_ceiling=estimate_ceiling)
#
#    @classmethod
#    def get_norm_weight(cls, qwt, cwt, bm25_ceiling=100, estimate_ceiling=False):
#        qw_len = len(qwt.get_all_weights())
#        qw_non_zero = len(list(filter(lambda k: k > 0, qwt.get_all_weights())))
#
#        if estimate_ceiling:
#            bm25_ceiling = qw_non_zero / qw_len * bm25_ceiling
#
#        cosine_ceiling = len(list(filter(lambda k: k > 0, cwt.get_all_weights())))
#
#        # Analytical solution for getting log base:
#        # n_score - log(bm25_score)/log(x) = 0
#        # Solve for x
#        return bm25_ceiling ** (1 / float(cosine_ceiling))
#