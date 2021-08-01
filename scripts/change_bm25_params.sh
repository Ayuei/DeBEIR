#!/bin/bash

k="1.4"
b="0.9"

curl -X POST "localhost:9200/test_trials/_close?wait_for_active_shards=0&pretty"
curl -X PUT "localhost:9200/test_trials/_settings?pretty" -H 'Content-Type: application/json' -d"
{
  \"index\": {
    \"similarity\": {
      \"default\": {
        \"type\": \"BM25\",
        \"k1\": ${k},
        \"b\": ${b}
      }
    }
  }
}
"
curl -X POST "localhost:9200/test_trials/_open?pretty"