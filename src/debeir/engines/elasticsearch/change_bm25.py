import json

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


def change_bm25_params(index, k1: float, b: float, base_url: str="http://localhost:9200"):
    """
    Change the BM25 parameters of the elasticsearch BM25 ranker.

    :param index: The elasticsearch index name
    :param k1: The k parameter for BM25 (default 1.2) [Usually 0-3] [Term saturation constant] ->
               The higher the k value, the more weight given to document that repeat terms.
    :param b: The b parameter for BM25 (default 0.75) [Usually 0-1] [Document length constant] ->
              The higher the b value, the higher it penalises longer documents.
    :param base_url: The elasticsearch base URL for API requests (without index suffix)
    """
    base_url = f"{base_url}/{index}"

    resp = requests.post(base_url + "/_open?pretty", timeout=60)

    if not resp.ok:
        raise RuntimeError("Response code:", resp.status_code, resp.text)

    resp = requests.post(base_url + "/_close?pretty", timeout=60)

    if not resp.ok:
        raise RuntimeError("Response code:", resp.status_code, resp.text)

    headers = {"Content-type": "application/json"}

    data = {
      "index": {
        "similarity": {
          "default": {
            "type": "BM25",
            "b": b,
            "k1": k1,
          }
        }
      }
     }

    resp = requests.put(base_url+"/_settings", headers=headers, data=json.dumps(data), timeout=60)

    if not resp.ok:
        raise RuntimeError("Response code:", resp.status_code, resp.text)

    resp = requests.post(base_url + "/_open?pretty", timeout=60)

    if not resp.ok:
        raise RuntimeError("Response code:", resp.status_code, resp.text)
