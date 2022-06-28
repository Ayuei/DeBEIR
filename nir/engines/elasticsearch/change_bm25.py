import elasticsearch
import requests
from loguru import logger

# echo "k = $k b = $b"
#
# curl -X POST "localhost:9200/${INDEX}/_close?pretty"
#
# curl -X PUT "localhost:9200/${INDEX}/_settings?pretty" -H 'Content-Type: application/json' -d"
# {
#  \"index\": {
#    \"similarity\": {
#      \"default\": {
#        \"type\": \"BM25\",
#        \"b\": ${b},
#        \"k1\": ${k}
#      }
#    }
#  }
# }"
# curl -X POST "localhost:9200/${INDEX}/_open?pretty"
#
# sleep 10


def change_bm25_params(client: elasticsearch.Elasticsearch, k1: float, b: float):
    """
    Change the BM25 parameters of the elasticsearch BM25 ranker.

    :param client: The elasticsearch client object
    :param k1: The k parameter for BM25 (default 1.2) [Usually 0-3] [Term saturation constant] ->
               The higher the k value, the more weight given to document that repeat terms.
    :param b: The b parameter for BM25 (default 0.75) [Usually 0-1] [Document length constant] ->
              The higher the b value, the higher it penalises longer documents.
    """
    logger.info("blah")
    base_url = "localhost:9200/{index}"

    resp = requests.get(base_url + "/_close?pretty")

    if not resp.ok:
        raise RuntimeError("Response code:", resp)

    headers = {"Content-type": "application/json"}
    data = """"{
      "index": {
        "similarity": {
          "default": {
            "type": "BM25",
            "b": %f,
            "k1": %f, 
          }
        }
      }
     }""" % (
        k1,
        b,
    )

    resp = requests.get(base_url, headers=headers, data=data)

    if not resp.ok:
        raise RuntimeError("Response code:", resp)

    resp = requests.get(base_url + "/_close?pretty")

    if not resp.ok:
        raise RuntimeError("Response code:", resp)
