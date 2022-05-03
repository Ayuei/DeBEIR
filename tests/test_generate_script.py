import pprint
from unittest import TestCase
from query import generate_script as gs


class Test(TestCase):
    def test_generate_script(self):
        q = "query_x"
        question = "question_x"
        narr = "narrative_x"
        q_eb = [0]
        qstn_eb = [1]
        narr_eb = [2]
        weights = [1,2,3,4,5,6,7,8,9]
        params = {
         "q_eb": q_eb,
         "qstn_eb": qstn_eb,
         "narr_eb": narr_eb,
         "weights": weights,
         # "reduce_offset": len(cosine_weights) - sum(cosine_weights),
         "norm_weight": 1,
         "divisor": 1.0,
         "offset": 1.0,
         "disable_bm25": False
        }

        script = gs.generate_script(
            fields=[q,question,narr],
            params=params,
            source_generator=gs.generate_source,
        )

        script2 = generate_query(q, question, narr, q_eb, qstn_eb, narr_eb, norm_weight=1)

        pprint.pprint(script, open("blah.txt", "w+"))
        pprint.pprint(script2, open("blah2.txt", "w+"))

        assert script == script2

def generate_query(
    q,
    qstn,
    narr,
    q_eb,
    qstn_eb,
    narr_eb,
    cosine_weights=[1] * 9,
    query_weights=[1] * 12,
    expansion="disease severe acute respiratory syndrome coronavirus treatment virus covid-19 sars-cov-2 covid sars",
    norm_weight=2.15,
    disable_bm25=False,
):
    assert len(query_weights) == 12
    assert len(cosine_weights) == 9

    expansion = ""  # set expansion to nothing for submission

    return {
        "_source": {
            "excludes": ["*.abstract_embedding_array", "*.fulltext_embedding_array"]
        },
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        # Match on title, abstract and fulltext on all three fields
                        # Weights should be added later
                        "should": [
                            {
                                "match": {
                                    "title": {"query": q, "boost": query_weights[0]}
                                }
                            },
                            {
                                "match": {
                                    "title": {"query": qstn, "boost": query_weights[1]}
                                }
                            },
                            {
                                "match": {
                                    "title": {"query": narr, "boost": query_weights[2]}
                                }
                            },
                            {
                                "match": {
                                    "title": {
                                        "query": expansion,
                                        "boost": query_weights[3],
                                    }
                                }
                            },
                            {
                                "match": {
                                    "abstract": {"query": q, "boost": query_weights[4]}
                                }
                            },
                            {
                                "match": {
                                    "abstract": {
                                        "query": qstn,
                                        "boost": query_weights[5],
                                    }
                                }
                            },
                            {
                                "match": {
                                    "abstract": {
                                        "query": narr,
                                        "boost": query_weights[6],
                                    }
                                }
                            },
                            {
                                "match": {
                                    "abstract": {
                                        "query": expansion,
                                        "boost": query_weights[7],
                                    }
                                }
                            },
                            {
                                "match": {
                                    "fulltext": {"query": q, "boost": query_weights[8]}
                                }
                            },
                            {
                                "match": {
                                    "fulltext": {
                                        "query": qstn,
                                        "boost": query_weights[9],
                                    }
                                }
                            },
                            {
                                "match": {
                                    "fulltext": {
                                        "query": narr,
                                        "boost": query_weights[10],
                                    }
                                }
                            },
                            {
                                "match": {
                                    "fulltext": {
                                        "query": expansion,
                                        "boost": query_weights[11],
                                    }
                                }
                            },
                        ],
                        "filter": {"range": {"date": {"gte": "2019-12-31"},}},
                    }
                },
                "script": {
                    "lang": "painless",
                    # Compute dotproducts as some Vectors are zero vectors
                    # Use dotproducts as a proxy to see if we're able to compute the cosine similarity
                    # Otherwise return 0
                    # We have to do this as the values of Vectors in elasticsearch are not only
                    # PRIVATE but ALSO encoded in BINARY that non-trivally decoded.
                    # Weights should be added later
                    "source": """
                               def weights = params.weights;
                               // If document score is zero, don't do score calculation
                               // Filter query has set it to zero
                               if (Math.signum(_score) == 0){
                                   return 0.0;
                               }

                               if (params.norm_weight < 0.0) {
                                   return _score;
                               }

                               double q_t = dotProduct(params.q_eb, 'title_embedding');
                               double qstn_t = dotProduct(params.qstn_eb, 'title_embedding');
                               double narr_t = dotProduct(params.narr_eb, 'title_embedding');

                               double q_abs = dotProduct(params.q_eb, 'abstract_embedding');
                               double qstn_abs = dotProduct(params.qstn_eb, 'abstract_embedding');
                               double narr_abs = dotProduct(params.narr_eb, 'abstract_embedding');

                               //double q_tb = 0.0;
                               //double qstn_tb = 0.0;
                               //double narr_tb = 0.0;

                               //try{
                               //     q_tb = dotProduct(params.q_eb, 'fulltext_embedding');
                               //     qstn_tb = dotProduct(params.qstn_eb, 'fulltext_embedding');
                               //     narr_tb = dotProduct(params.narr_eb, 'fulltext_embedding');
                               // } catch(Exception e){
                               // }

                               if (Math.signum(q_t) != 0){
                                   q_t = weights[0]*cosineSimilarity(params.q_eb, 'title_embedding') + params.offset;
                               }

                               if (Math.signum(qstn_t) != 0){
                                   qstn_t = weights[1]*cosineSimilarity(params.qstn_eb, 'title_embedding') + params.offset;
                               }

                               if (Math.signum(narr_t) != 0){
                                   narr_t = weights[2]*cosineSimilarity(params.narr_eb, 'title_embedding')+params.offset;
                               }

                               if (Math.signum(q_abs) != 0){
                                   q_abs = weights[3]*cosineSimilarity(params.q_eb, 'abstract_embedding')+params.offset;
                               }

                               if (Math.signum(qstn_abs) != 0){
                                   qstn_abs = weights[4]*cosineSimilarity(params.qstn_eb, 'abstract_embedding')+params.offset;
                               }

                               if (Math.signum(narr_abs) != 0){
                                   narr_abs = weights[5]*cosineSimilarity(params.narr_eb, 'abstract_embedding')+params.offset;
                               }

                               //if (Math.signum(q_tb) != 0){
                               //    q_tb = weights[6]*cosineSimilarity(params.q_eb, 'fulltext_embedding')+1.0;
                               //}

                               //if (Math.signum(qstn_tb) != 0){
                               //    qstn_tb = weights[7]*cosineSimilarity(params.qstn_eb, 'fulltext_embedding')+1.0;
                               //}

                               //if (Math.signum(narr_tb) != 0){
                               //    narr_tb = weights[8]*cosineSimilarity(params.narr_eb, 'fulltext_embedding')+1.0;
                               //}

                               // return q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs + Math.log(_score)/Math.log(1.66); // 2.15
                               // return (q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs)/params.divisor + Math.log(_score+1)/Math.log(params.norm_weight); // 2.15 // 1.66
                               // return Math.log(_score+1)/Math.log(params.norm_weight);
                               // return (q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs)/params.divisor;

                               // return _score;
                               double score = 0.0;

                               if (params.disable_bm25 == true) {
                                  score = 0.0;
                               } else {
                                  score = Math.log(_score)/Math.log(params.norm_weight);
                               }

                               return q_t + qstn_t + narr_t + q_abs + qstn_abs + narr_abs - params.reduce_offset + score; // 2.15
                               """,
                    "params": {
                        "q_eb": q_eb,
                        "qstn_eb": qstn_eb,
                        "narr_eb": narr_eb,
                        "weights": cosine_weights,
                        "reduce_offset": len(cosine_weights) - sum(cosine_weights),
                        "norm_weight": norm_weight,
                        "divisor": 1.0,
                        "offset": 1.0,
                        "disable_bm25": disable_bm25,
                    },
                },
            }
        },
    }
