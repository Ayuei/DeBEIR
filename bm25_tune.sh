#!/bin/sh
set -e

for k in $(seq 0.2 0.1 3); do
	for b in $(seq 0 0.1 1); do
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
		python main.py --addr "localhost" \
			       --es-port 9200 \
			       --topics-path ./assets/2015_descriptions.txt \
			       --index-name "test_trials" \
			       --output-file "./outputs/bm25_tuning/${k}_${b}.txt" \
			       --size 1000
	done
done
