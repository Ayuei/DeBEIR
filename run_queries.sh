#!/bin/sh

source venv/bin/activate

python main.py --addr "localhost" \
	       --es-port 9200 \
	       --topics-path ./assets/2015_descriptions.txt \
	       --index-name "test_trials" \
	       --output-file "./outputs/tuned2.txt" \
	       --size 1000
