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


def change_bm25_params(client: elasticsearch.Elasticsearch, k: float, b: float):
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
        k,
        b,
    )

    resp = requests.get(base_url, headers=headers, data=data)

    if not resp.ok:
        raise RuntimeError("Response code:", resp)

    resp = requests.get(base_url + "/_close?pretty")

    if not resp.ok:
        raise RuntimeError("Response code:", resp)
