#!/bin/sh

source venv/bin/activate

for i in $(seq 23 102); do
	python ablations.py --addr "localhost" \
	       	    --es-port 9200 \
	       	    --topics-path ./assets/2015_descriptions.txt \
	       	    --index-name "test_trials" \
		    --mapping-id $i \
	       	    --output-file "./outputs/ablations.txt"
done
